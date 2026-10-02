"""Characterization pins for ``eval/vision_rebase.py::main()`` (Phase 5.0 safety net).

``main()`` is a 2.8k-line orchestration that, until now, was pinned only through its leaf helpers.
This file drives the *real* ``main()`` end to end on a tiny offline world and pins everything it
produces, so that every later move out of ``vision_rebase.py`` is checked at hash level:

* the final summary (``run_logger.log_summary``), via ``_hashing.hash_json`` (timing / memory /
  path / git / dataset-identity keys masked; tmp-dir prefixes normalised);
* every ``.pt`` file written under the run directory (``hash_tensor_dict``);
* the ``resolved_config`` handed to ``start_run`` and the sequence of ``log_event`` calls;
* the sorted tree of summary key paths, event names and saved-file names
  (``main_golden_structure.json``).

What is faked (patched BY NAME across every module in ``PATCH_MODULES`` so the harness survives the
Phase 5 moves; ``raising=False`` makes names that a module does not import yet harmless):
``OpenClipClassifier`` (tiny real attention ViT-like ``_AttnModel``), ``load_ckpt`` /
``resolve_ckpt_path`` (in-memory seeded perturbations of the base state dict), ``SUITES`` /
``load_hf_splits`` / ``extract_classnames`` / ``build_vision_loaders`` /
``build_vision_calibration_loader`` (two-task tiny seeded class loaders), ``eval_task_top1`` (a real,
weight-sensitive score on fixed tensors) and ``start_run`` (a recorder). Everything else -- config
resolution, alpha search, BRACE, THESEUS / BiCo / Ariadne, the merge registry, task vectors,
``torch.save`` -- is the real code.

Regenerating (only after establishing that a change is intended; see HASHES.md):
``GOLDEN_CAPTURE=/path/out.txt GOLDEN_CAPTURE_STRUCTURE=tests/golden/main_golden_structure.json
pytest tests/golden/test_main_golden.py -q -p no:cacheprovider``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ._hashing import deterministic_cpu, hash_json, hash_tensor_dict

# Modules whose namespaces get the fakes. Add the Phase 5 modules here as they appear; names a module
# does not import are simply created on it (``raising=False``), which is harmless.
PATCH_MODULES: list[str] = [
    "merge_and_rebase.eval.vision_rebase.cli",
    "merge_and_rebase.eval.vision_rebase.source_lmc",
    "merge_and_rebase.eval.vision_rebase.context",
    "merge_and_rebase.eval.vision_rebase.merge",
    "merge_and_rebase.eval.vision_rebase.pipeline",
    "merge_and_rebase.eval.vision_rebase.alpha_search",
    "merge_and_rebase.eval.vision_rebase.stages",
    "merge_and_rebase.eval.vision_rebase.completion",
    "merge_and_rebase.eval.vision_rebase.method_stages",
    "merge_and_rebase.rebase.orchestration",
]

STRUCTURE_PATH = Path(__file__).with_name("main_golden_structure.json")

SRC_TAG = "fake_source"
TGT_TAG = "fake_target"
N_CLASSES = 5
IN_DIM = 6
TASKS = ("MNIST", "DTD")  # real suite task names: main() calls get_templates(task)

EXPECTED: dict[str, str] = {
    "ariadne_calibration_tiny_imagenet:events": "01db443136878053b9e90896aedd6077dd33f7438bbcf7aa0cb0240016851fef",
    "ariadne_calibration_tiny_imagenet:resolved_config": "1ccc6ac0ce56a103ba8fc2114f6104d4ff6b472fe9e53610a7eda1e4b16a47f0",
    "ariadne_calibration_tiny_imagenet:summary": "4dc908ea937fd52a40c32b1aae0b4ed98fdeff14654bf33bffd3592bb34f0e6d",
    "ariadne_calibration_vision8_mix:events": "5694dcbe0ff533debe2db4226ba8f14b7dca3f1259c81b1b19e2cb0c47ddaa3b",
    "ariadne_calibration_vision8_mix:resolved_config": "89440adafe07e40054b0a54f8275f164db334da6f6b596fd289ce7668168794a",
    "ariadne_calibration_vision8_mix:summary": "e02f85eca46abb016059a0da7de6c2a21b7f3978cc6903e198ba90e8c2ac0dda",
    "ariadne_merge_in_source_then_fit:events": "bffa806a91355fd50e6a5d73a908ed2c717e2fd009e98f6102908d6641ecb8b0",
    "ariadne_merge_in_source_then_fit:resolved_config": "a13dc4a3d22d1fb4525970f7abfc022f0dc86fdc07512e5321c9695466d13c10",
    "ariadne_merge_in_source_then_fit:summary": "50d5ebb3b8b1e5ec5bb138d15d93b13d6d8c1a386e3066b6979f7da55ac06334",
    "ariadne_spelling_ariadne:events": "c690c559481aa8f7fbaf38ac0079ad75bb290d4096f445e32e4f5cd370027c1c",
    "ariadne_spelling_ariadne:file:tvs/DTD_ariadne_transported_native.pt": "1d6fdae2a1278fff1640fe7e14a525e42a1ebfdc7cb33f9b8106de02065d609e",
    "ariadne_spelling_ariadne:file:tvs/MNIST_ariadne_transported_native.pt": "3724cf5d80fb125e723881e9c5bbe5fe48f4e0be35df60dbd65f51ad51f60c60",
    "ariadne_spelling_ariadne:resolved_config": "1f1689e9a2421fabffe7f62929bd966a22872a1553658e28653b1f835c424624",
    "ariadne_spelling_ariadne:summary": "7be896b51f45cb5b2a34fd2116a174127b012531760e932d48aebfefdaff09c6",
    "ariadne_spelling_direct_residual:events": "b37f66e981c3771d4d73650987dbbcfdf789e72aa63ca71ffc017f1764add678",
    "ariadne_spelling_direct_residual:resolved_config": "37492df2cb806d9d46ad224ec71febb60b6e7081c79eeae57d62f3e6b30d6412",
    "ariadne_spelling_direct_residual:summary": "d801522a1299cc04386b066a5f6ffdb7563537f6d883a9f75ebc25df7a9f4b7b",
    "bico_extend_depth_alignment_absent:events": "2951ff3e64243888e29519fe102b43f475b518b670531d8efb71e70c7691266d",
    "bico_extend_depth_alignment_absent:resolved_config": "1cc8de73ef3df725f04a51f15dc29104eccce03f77cab8f8af50eceea9fafa28",
    "bico_extend_depth_alignment_absent:summary": "4c0318fa57e3a3a55cf97339e50eb8676876304499f7abedffd029836791b6e4",
    "bico_extend_depth_defaults_method:events": "76e14d3a2f64cb79c72bada854eb6e4d744788907a8603b1fa09a1892effaf8c",
    "bico_extend_depth_defaults_method:resolved_config": "562634f183fa57a07af0b338b392df616303ea1d42d028efee84421d150dfee2",
    "bico_extend_depth_defaults_method:summary": "71254273e44f218f8e4ed9b7e3c36e5de6eaec244dbed846812e1c20328f6481",
    "bico_extend_discrete_index_match:events": "76e14d3a2f64cb79c72bada854eb6e4d744788907a8603b1fa09a1892effaf8c",
    "bico_extend_discrete_index_match:resolved_config": "4d6b1f67141042fc05f253362a54d1884a1853194a34e45041a2b5cdad710e32",
    "bico_extend_discrete_index_match:summary": "2cf07e134ce9bbb48fe981ae7ad4906ada91324dae63226caba2d8272305bb49",
    "direct_residual_sequential_endpoints_load:events": "1ef62c7535846934d8045d5edd63b74276939b40ba01658594f6d93dc4363789",
    "direct_residual_sequential_endpoints_load:resolved_config": "4bc7bb4e4aadc1241a58c6471938f47ac8f50ec3ccef2bd1ffd8fbecdb2a9a0a",
    "direct_residual_sequential_endpoints_load:summary": "d5c70d0c65668395e79a233e1f56b050352e656cc4a6fe67981afe3e512b56ff",
    "direct_residual_sequential_endpoints_save:events": "1ef62c7535846934d8045d5edd63b74276939b40ba01658594f6d93dc4363789",
    "direct_residual_sequential_endpoints_save:file:tvs/DTD_direct_residual_transported_native.json": "9c8385b01ce5006b4c2cf26f7dd43c35da27cfdfd8d2d597a3e642c0c4a4fad9",
    "direct_residual_sequential_endpoints_save:file:tvs/DTD_direct_residual_transported_native.pt": "3b71fbd5746bbf862be1a5423b90395a1c06364bc9390d887febba44dab885f8",
    "direct_residual_sequential_endpoints_save:file:tvs/MNIST_direct_residual_transported_native.json": "1c4e6dee06f77b8163b146c6909f9583c1c2caf4093e1678250525fd1a71082a",
    "direct_residual_sequential_endpoints_save:file:tvs/MNIST_direct_residual_transported_native.pt": "82292423e9736ec1d41862a788f9c4771386fcea76ab7c83a73e3835cefcf244",
    "direct_residual_sequential_endpoints_save:resolved_config": "4bc7bb4e4aadc1241a58c6471938f47ac8f50ec3ccef2bd1ffd8fbecdb2a9a0a",
    "direct_residual_sequential_endpoints_save:summary": "21047a411b7d7faccf11123b992eee3194ea4c8e0c23fe8edee15b4b2196a11b",
    "gradfix_same_architecture:events": "9488b1ca2ed5df214e447323d9eb4d903743de9cb480a72eb0d36003488fa696",
    "gradfix_same_architecture:resolved_config": "009d49f15cc0c5a9558e093cd81da6345911f241c2621acf7de9111b9724afb4",
    "gradfix_same_architecture:summary": "3fcac87bf3e96cf8bd7a12a22cb7324d74946f38f505172463c35bb47fee17f5",
    "theseus_alpha_patience_per_task:events": "0dabeb6947b402ee9e99bf244c6bcbcb940a0cd1d4334481bbb47dcf93bbfab8",
    "theseus_alpha_patience_per_task:resolved_config": "39c946ec679dd6001390c83d96c18e7ae47ba5789c8d1a0a7106b544d9fe9160",
    "theseus_alpha_patience_per_task:summary": "61369c10bdc122ec53c5e6a108e80489e0730a2ea9902ea08ab5086b53076828",
    "theseus_alpha_patience_shared:events": "153553151c33d17a5d6319865dce70af827f93a7a38704c1233ddc0018c0caa3",
    "theseus_alpha_patience_shared:resolved_config": "40461ceb706dd5957ed2127c293402d4af84a44c6486f272a4a03a3ac5440497",
    "theseus_alpha_patience_shared:summary": "cdc5f15a24272fef41e04c34d1c32b1b48617569bfaf3ee3f27d50da4c9f0931",
    "theseus_brace_merge_then_transport:events": "d7c196e1874ada01e3a8f2d303b63a7b99ccf5f1476120652951bba8d70b0c5c",
    "theseus_brace_merge_then_transport:resolved_config": "9641014779be7127e747c89af3122e646c2c7a00cbdd2ab0d1671ad83f7c8489",
    "theseus_brace_merge_then_transport:summary": "f91ba010e6afa7aeeaa64ee409147d653bf9639f41d050c625d4ceb3b189ae07",
    "theseus_double_direct_p1_correction:events": "9125fc408ba8d27d990778e803ecc35aa043088d74baabf43fad70129d566b16",
    "theseus_double_direct_p1_correction:resolved_config": "90ad0104b8143593654b8bf6cf3fe012bb8c06215308d71cbf9f944d0d9f3b54",
    "theseus_double_direct_p1_correction:summary": "fc20517084ddcaac51e22babe37c22e8e1f37ac1541d00a81ffe4da0e8b10b03",
    "theseus_double_joint_blockwise_correction:events": "be17256a05c8956060e4e724c218d59f082baab38c493ffbd2ef70e898d7f068",
    "theseus_double_joint_blockwise_correction:resolved_config": "d378dcecc69804e3d0f6f6e617697af1aa7780064d3cbbb15d00b20f96685895",
    "theseus_double_joint_blockwise_correction:summary": "566e31e48d3e927784d0d149f929fdb6642a3c8bd60b7aca5b092a2738ac57fd",
    "theseus_double_target_residual_completion:events": "f5787332fff23f5a332d742ee95af50ce82efaeaeaaa06854a363e4e71c771a4",
    "theseus_double_target_residual_completion:resolved_config": "8dd4105595939ffd1b672210dc352735559f742940de7763844f429ff08fb6b4",
    "theseus_double_target_residual_completion:summary": "1f5baa798a0d7c0893dd903c07136511a928b62ff43a16da28f9504597345865",
    "theseus_double_target_residual_completion_direct_target:events": "f4b40b8b7479574858cc155c54fc5bced8c10feaedeef92e620664a1a7488e6f",
    "theseus_double_target_residual_completion_direct_target:resolved_config": "6213075122cea432869c79b8b696ea902041a893d0c7bea0ee23c925246af280",
    "theseus_double_target_residual_completion_direct_target:summary": "5066935f322770dc6b73fbf90cfac957fe0789ff1bc4978076e2c9f595be0dec",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:events": "770993575bb34821bd491cf840365d33698cd7ae31d35478f427023f278f2c34",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:file:tvs/DTD_theseus_transported_legacy_visual.pt": "2b2f8a33258f9b626bc98ad9096c084b30caf5f1d8f1e856c34e8f4070fe9592",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:file:tvs/DTD_theseus_transported_legacy_visual_no_conv1.pt": "2b2f8a33258f9b626bc98ad9096c084b30caf5f1d8f1e856c34e8f4070fe9592",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:file:tvs/DTD_theseus_transported_native.pt": "4d0c4b2db99e7330fca87356e2ee912379bba820713430ddf24c4a63527b1f8a",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:file:tvs/MNIST_theseus_transported_legacy_visual.pt": "5b7931eeeee0dc23fc5dd0e573f21edb5af1fa6a7b9a82c40b479de97e733a87",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:file:tvs/MNIST_theseus_transported_legacy_visual_no_conv1.pt": "5b7931eeeee0dc23fc5dd0e573f21edb5af1fa6a7b9a82c40b479de97e733a87",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:file:tvs/MNIST_theseus_transported_native.pt": "5380b94239486f7d90652b2412c72547ef59d4c41aab22d0c68f73a28a50797b",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:resolved_config": "4e48f4f00d41c7c2b18f7f30a0774137e6116b274ba1dbec1acdd267b557fdaa",
    "theseus_equal_depth_none_fixed_alpha_save_tvs:summary": "bb413791fe0a4769227075222585b2c73ea33e5e64045b947602654dc4d31ce0",
    "theseus_extend_brace_correction_search_shared:events": "33ed39db4f073498b30cf8450f95703fbb34a17b2dd4cffdcaeb574431e0bc4a",
    "theseus_extend_brace_correction_search_shared:resolved_config": "1d801ce725017a23c49aeef6622822c117346f6e7fe8ef4c22f3eae5cd815230",
    "theseus_extend_brace_correction_search_shared:summary": "7209af874be8b0d3f9c0698f788116a4a6b3247738e6f69ee74320cfc7775dfe",
    "theseus_extend_brace_skip_correction:events": "4d941430e9502494ada076b4a87e4110c284be813f4127d263f7ccf848608a8e",
    "theseus_extend_brace_skip_correction:resolved_config": "1439398ce009ea207cbf1265acb38c618f1a3e4c531aaf7b29b2353c452f11e4",
    "theseus_extend_brace_skip_correction:summary": "5de3ef413ff1866bbdc1dac7d45edbdf90b1d24c2b9d96dab44c5f973c1f6c61",
    "theseus_extend_depth_defaults_method:events": "4d941430e9502494ada076b4a87e4110c284be813f4127d263f7ccf848608a8e",
    "theseus_extend_depth_defaults_method:resolved_config": "b421f8e9b43121622c0f79d27bf09975c59ea00fa1d07b0e5ddb2c49def2d0cb",
    "theseus_extend_depth_defaults_method:summary": "5de3ef413ff1866bbdc1dac7d45edbdf90b1d24c2b9d96dab44c5f973c1f6c61",
    "theseus_extend_eval_before_and_source_lmc:events": "78e670a4a3c3c01ff789067d482f2ce62ac4fa44927f02f5548589aa4582c5db",
    "theseus_extend_eval_before_and_source_lmc:resolved_config": "11cbfbfb16b0e070f832dd985939c2ebd87945fac802abb3cc234fcf9d0e9d3c",
    "theseus_extend_eval_before_and_source_lmc:summary": "0dacee762c854d324794899a6fd177df67b19222ccaa6dd1045576148ccb1bd4",
    "theseus_independent_endpoint_average:events": "a7e42d55fdcc06a546c261a49198eb289a5f6021ec72eb4a6eaa2e05ee2519a4",
    "theseus_independent_endpoint_average:resolved_config": "ae31cf15dd14ec899f2f2dcba3fee516f411d266a1ec07312074ed9ec292653f",
    "theseus_independent_endpoint_average:summary": "445bb0c91dd6c97cf3c758088276c0ef47a617cd69846e66a084260779d43abd",
    "theseus_merge_then_brace_then_transport_correction:events": "781544f517b2ba512ea1697c9d58115dfadac20b1b7a0204403f6e8a0307428d",
    "theseus_merge_then_brace_then_transport_correction:resolved_config": "4ca7dce6ff0a9b5262edb18688b55b9ec71e8fde885ff9c99c05203bf168d0b7",
    "theseus_merge_then_brace_then_transport_correction:summary": "27a848588619005a12f7047366ec8f1a262493f1c4f1d4af85a2f081e2407fc3",
    "theseus_merge_then_brace_then_transport_skip_correction:events": "a1aa7dae60d1ca1568f590a7307f1e0bc2e5c315f499d7de85f651bf2b2c2946",
    "theseus_merge_then_brace_then_transport_skip_correction:resolved_config": "5ab34a672ef40da2114a0a20f3475f4981870025e95427c28a69a19b2f540c70",
    "theseus_merge_then_brace_then_transport_skip_correction:summary": "91663584a709e6f11f6b2cb1504f12ee3fc289b32845c4b1fcfe37955960b9b8",
    "theseus_merge_then_rebase:events": "7728a27a5d763b3c62d138f397d975ecbeecf289a4d6abba4631f7ef6d8d78e7",
    "theseus_merge_then_rebase:resolved_config": "5347bf611bb740e75c4b86c6e3a293102ca6a72bfa5630042f0241015ddef240",
    "theseus_merge_then_rebase:summary": "4f82c86e5e3b860090df80378454fb6ad50e24cf1c11bf22467a47985eef2a0e",
    "theseus_merge_then_rebase_tiny_protocol:events": "b121b28fec53012d78d40af57c2d7ffb11eb946899994ef6be36e56f1888cb3c",
    "theseus_merge_then_rebase_tiny_protocol:resolved_config": "a604f1007cc4d329527a702b72313441afb2f288a4bd3c85274ba6f752e51cde",
    "theseus_merge_then_rebase_tiny_protocol:summary": "085b8e58641f64593e1c03a5febeccff19100bd7d5ef0d6d93ca0c8a81c26d52",
    "theseus_native_target_auto_detected_per_task:events": "a905c680f869e7087bd2aa0fc353928a4bec6d96c8f1f30e0d85d5f41612eecf",
    "theseus_native_target_auto_detected_per_task:resolved_config": "38ea5fa0a04bd9dc5668a9e75c69ef7169ae3e9caf58ade2ecde22b26eb01f46",
    "theseus_native_target_auto_detected_per_task:summary": "40c9adb678a3b0146f089f33b77a1002ef7e9306e00bff8ba00527fa468e6e03",
    "theseus_native_target_explicit:events": "07e139c8ba220ebf927705756b5c3315c498f017ea48c53acef0720a4e6764aa",
    "theseus_native_target_explicit:resolved_config": "9a8a82ecbc021c294c82cc76762276d99e8596baa31fa245c5f5ee0bac7b6173",
    "theseus_native_target_explicit:summary": "7aabe30bf39be00aea4710091163ff15e377a7d7d6ec5eec8bbe2998aa49539f",
    "theseus_rebase_then_merge_per_task_hierarchical:events": "03465f70d0ecd7ae22fb9b0a830e0c7f821fbde6bf383c149cfa4d8a6963db6a",
    "theseus_rebase_then_merge_per_task_hierarchical:resolved_config": "1353eeec7eaf5e4e4c815118f2fa527b82e8e8558187296bf180360a68a6c487",
    "theseus_rebase_then_merge_per_task_hierarchical:summary": "a23bc0596002a599087c3c0496500c2bfd0e6233df3ffd4783da082671bfec38",
    "theseus_rebase_then_merge_per_task_no_global_search:events": "892fcba861cc7d56d5f8fecda5847a8cb161b66eccef2451ea28c4bcaa9e1e39",
    "theseus_rebase_then_merge_per_task_no_global_search:resolved_config": "e15458047adb2ce4e54c3219b4931bc211623ac7ae19fcb9bff4b6ff0c419cde",
    "theseus_rebase_then_merge_per_task_no_global_search:summary": "8c204d99477751c5660dae497a189afab630d19f8575a4def3880b3bde02b110",
    "theseus_rebase_then_merge_save_merged:events": "6778e5548a1a14aaa449bd686e62f7f8d647aff6531077e6a569da0695ddcd75",
    "theseus_rebase_then_merge_save_merged:file:merged/merged.pt": "44dc512d001dd7c96020b2244ccbaf8845111b6b5fff304c191cfc3968ccb3c6",
    "theseus_rebase_then_merge_save_merged:resolved_config": "60a99991b1828b5c6d88b24c66ea597feaf9166fdd2c0713b8e5732db81a7da6",
    "theseus_rebase_then_merge_save_merged:summary": "5f16ae0f844e00da4a9b9a7d906819cef00a4f5342f8f92707be2d2806c6afd0",
    "theseus_rebase_then_merge_shared:events": "6778e5548a1a14aaa449bd686e62f7f8d647aff6531077e6a569da0695ddcd75",
    "theseus_rebase_then_merge_shared:resolved_config": "e9900746e468c5f960d7f34f10493c4188f1a7b50c010a45368115a9cc588a00",
    "theseus_rebase_then_merge_shared:summary": "5f16ae0f844e00da4a9b9a7d906819cef00a4f5342f8f92707be2d2806c6afd0",
    "theseus_same_depth_direct_target:events": "a90dec52e9f2d9e3dbeb7acf1905fcfbd735233458cde2890d1121eafe3c2e47",
    "theseus_same_depth_direct_target:resolved_config": "a09a19560b04945b58a68f52889e95eb4edf2daa23b0625a21cbbc8b7fdf245e",
    "theseus_same_depth_direct_target:summary": "fe5d459a467b0926b559ee22bd8f874e9b4d4a6a3156661c03eab172c5cd9330",
    "theseus_samearch_none_alpha_search_untransported:events": "033dc0d386b3870ff1b26d7d5eca56c0b5cde001d374f24ef22f16cb26e1d96b",
    "theseus_samearch_none_alpha_search_untransported:resolved_config": "2c425532a53836dd72b074b2fd74d95dcfcfc92b91049f827f403a30b55f98ec",
    "theseus_samearch_none_alpha_search_untransported:summary": "e5eee073392d69dc0c96500bcd5862b118a7832f2a4dc053589e1822968abc89",
    "theseus_samedepth_eval_before_rebase:events": "2f94eff395370b1335e112bc692f82d9d3320f113b8212717aae0ea07259b3d7",
    "theseus_samedepth_eval_before_rebase:resolved_config": "c6c470d2cd9539809f274b146fb6c4a0a839415c7f165a4f87b50075b8ada306",
    "theseus_samedepth_eval_before_rebase:summary": "fb5708f4702f9fbb6953c9ac00fc6eb3f0b08a4680e08f2560c69798be97d7ce",
    "theseus_shrink_brace:events": "0370b172ea2a3d7c13dec63092774919e047a89f43e3106ef4f275429a8b7b40",
    "theseus_shrink_brace:resolved_config": "60ca4c7c30730404763a1a3f2e04687480ca58d04c1e447907732afb0a488d88",
    "theseus_shrink_brace:summary": "39afe6a1ddc9ddb9191f2a23d5d1036f1975991db5d46c97a65b45ef0abb0df5",
    "theseus_shrink_depth_defaults_method:events": "f5a2c6855e824ccd1767b9bf23dc9e2c02652965546ac3c069d9818929e19b79",
    "theseus_shrink_depth_defaults_method:resolved_config": "b421f8e9b43121622c0f79d27bf09975c59ea00fa1d07b0e5ddb2c49def2d0cb",
    "theseus_shrink_depth_defaults_method:summary": "6668b1388aeb062c1ba7191db22146403084b0e76d82719e4c8c44c4d86d4b8e",
    "theseus_transport_calibration_tiny_imagenet:events": "026fce97657d7dc893ee946b46f8edb1755b3a66e6522c1bac60f18ab6fff16f",
    "theseus_transport_calibration_tiny_imagenet:resolved_config": "90d671f2c3f5ed16b1d11d9a0eabfe6a4ff2e2a17d28839569793228500d63a3",
    "theseus_transport_calibration_tiny_imagenet:summary": "9f5938f5e0aa698dc635c412ed726de7587b471e9d87f751808cc3fa2ef0be83",
}


@pytest.fixture(autouse=True)
def _deterministic():
    # main() flips process-global determinism flags (``_set_deterministic_seed``); restore them afterwards.
    prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        with deterministic_cpu(seed=0):
            yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev


# --------------------------------------------------------------------------------------
# Tiny model (copy of test_release_golden_hashes._AttnModel so these pins do not move
# with that file's fixtures).
# --------------------------------------------------------------------------------------


class _AttnBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, num_heads=2, batch_first=True)
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, 2 * width)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(2 * width, width)),
                ]
            )
        )

    def forward(self, x):
        normalized = self.ln_1(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.mlp(self.ln_2(x))


class _AttnVisual(torch.nn.Module):
    def __init__(self, depth, in_dim=IN_DIM, width=4, out_dim=N_CLASSES):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_AttnBlock(width) for _ in range(depth)])
        self.ln_post = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Linear(width, out_dim, bias=False)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1).repeat(1, 3, 1)
        for block in self.transformer.resblocks:
            x = block(x)
        return self.proj(self.ln_post(x).mean(dim=1))


class _AttnModel(torch.nn.Module):
    def __init__(self, depth, width=4):
        super().__init__()
        self.visual = _AttnVisual(depth, width=width)
        self.logit_scale = torch.nn.Parameter(torch.zeros(()))

    def encode_image(self, x):
        return self.visual(x)

    def state_dict(self, *args, **kwargs):
        # A real run holds the model on CUDA, where ``to_cpu_fp32(model.state_dict())`` is a copy. On CPU it
        # would alias the live parameters and a later ``load_into_model`` would silently mutate
        # ``target_base_sd`` (tripping main()'s "target base mutated" guard). Cloning reproduces the
        # copy semantics of the GPU runs these pins stand in for.
        sd = super().state_dict(*args, **kwargs)
        if args or kwargs:
            return sd
        return OrderedDict((k, v.detach().clone()) for k, v in sd.items())


# --------------------------------------------------------------------------------------
# Deterministic tiny data
# --------------------------------------------------------------------------------------


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _gen(*parts: Any) -> torch.Generator:
    return torch.Generator().manual_seed(_seed(*parts))


class _Ds(TensorDataset):
    """Tensor dataset with the ``sample_ids`` Ariadne's paired calibration compares across views."""

    def __init__(self, x, y, sample_ids):
        super().__init__(x, y)
        self.sample_ids = tuple(sample_ids)


_SPLIT_SIZES = {"train": 16, "val": 8, "test": 8}


def _split_dataset(task: str, split: str, tag: str, n: int | None = None) -> _Ds:
    """Images differ per view (``tag``) by a small seeded perturbation; labels / ids are view-independent."""
    n = _SPLIT_SIZES.get(split, 16) if n is None else n
    x = torch.randn(n, IN_DIM, generator=_gen(task, split, "images"))
    if tag == SRC_TAG:
        x = x + 0.1 * torch.randn(n, IN_DIM, generator=_gen(task, split, "view", tag))
    y = torch.randint(0, N_CLASSES, (n,), generator=_gen(task, split, "labels"))
    return _Ds(x, y, [f"{task}:{split}:{i}" for i in range(n)])


def _classnames(task: str) -> list[str]:
    return [f"{task.lower()}_class_{i}" for i in range(N_CLASSES)]


def _tag_of(preprocess: Any) -> str:
    return str(getattr(preprocess, "tag", TGT_TAG))


# --------------------------------------------------------------------------------------
# Fake classifier / world
# --------------------------------------------------------------------------------------


def _tiny_score(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, int]:
    probs = torch.softmax(logits.double(), dim=-1)
    hard = float((logits.argmax(dim=-1) == labels).sum().item())
    soft = float(probs[torch.arange(labels.numel()), labels].sum().item())
    return hard, soft, int(labels.numel())


class _FakeClassifier:
    """Stands in for ``OpenClipClassifier``: same constructor / method surface that ``main()`` touches."""

    def __init__(
        self, model, tokenizer=None, preprocess=None, train_preprocess=None, *, normalize=True, logit_scale=100.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.train_preprocess = train_preprocess if train_preprocess is not None else preprocess
        self.normalize = normalize
        self.logit_scale = float(logit_scale)
        self._zs: torch.Tensor | None = None

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def _compute_zeroshot_text_features(self, classnames, cfg):
        tag = _tag_of(self.preprocess)
        rows = [torch.randn(N_CLASSES, generator=_gen("text", tag, c)) for c in classnames]
        feats = torch.stack(rows, dim=0)
        return feats / feats.norm(dim=-1, keepdim=True)

    def build_zeroshot_text_features(
        self, classnames, cfg, *, cache_dir=None, cache_tag="zs_text", force_rebuild=False
    ):
        self._zs = self._compute_zeroshot_text_features(classnames, cfg)

    @torch.no_grad()
    def __call__(self, images):
        feats = self.model.encode_image(images)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        return 10.0 * feats @ self._zs.t()

    @torch.no_grad()
    def top1(self, loader, device):
        self.eval()
        hard = soft = 0.0
        total = 0
        for x, y in loader:
            h, s, n = _tiny_score(self(x), y)
            hard, soft, total = hard + h, soft + s, total + n
        # 0.999 * accuracy + 0.001 * mean probability of the true class: still a top-1 accuracy to the
        # alpha search, but a weight change that flips no prediction still changes the bits.
        return float(0.999 * hard / max(1, total) + 0.001 * soft / max(1, total))

    def top1_with_text_features(self, loader, *, device, text_features, expected_num_classes=None):
        prev = self._zs
        self._zs = text_features / text_features.norm(dim=-1, keepdim=True)
        try:
            return self.top1(loader, device)
        finally:
            self._zs = prev


@dataclass
class World:
    """Everything the fakes need: model architectures, tuned checkpoints, native tasks."""

    source: tuple[int, int] = (2, 4)  # (depth, width)
    target: tuple[int, int] = (2, 6)
    tasks: tuple[str, ...] = TASKS
    native: tuple[str, ...] = ()
    ckpt_scale: float = 0.05
    built: dict[str, Any] = field(default_factory=dict)
    ckpts: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    mutate_ckpt: dict[str, Any] = field(default_factory=dict)  # sanity tests: path -> fn(sd)

    def arch(self, tag: str) -> tuple[int, int]:
        return self.source if tag == SRC_TAG else self.target

    def build_model(self, tag: str) -> torch.nn.Module:
        depth, width = self.arch(tag)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed("model", tag, depth, width))
            return _AttnModel(depth, width=width)

    def build_classifier(self, cfg) -> _FakeClassifier:
        tag = SRC_TAG if cfg.pretrained == SRC_TAG else TGT_TAG
        model = self.build_model(tag)
        clf = _FakeClassifier(model=model, preprocess=SimpleNamespace(tag=tag), normalize=True, logit_scale=100.0)
        self.built[tag] = clf
        return clf

    def checkpoint(self, path: str) -> dict[str, torch.Tensor]:
        if path not in self.ckpts:
            task = path.split("://", 1)[1]
            tag = TGT_TAG if task in self.native else SRC_TAG
            sd = {k: v.detach().clone() for k, v in self.build_model(tag).state_dict().items()}
            gen = _gen("ckpt", task)
            for key in sorted(sd):
                if key.startswith("visual.") and sd[key].is_floating_point():
                    sd[key] = sd[key] + self.ckpt_scale * torch.randn(sd[key].shape, generator=gen)
            if path in self.mutate_ckpt:
                self.mutate_ckpt[path](sd)
            self.ckpts[path] = sd
        return {k: v.clone() for k, v in self.ckpts[path].items()}

    def tuned_ckpts(self) -> dict[str, str]:
        return {t: f"ckpt://{t}" for t in self.tasks}


def _fake_suite(world: World) -> dict[str, Any]:
    split_map = {"train": "train", "test": "test"}
    return {
        "fake2": SimpleNamespace(
            name="fake2",
            tasks=list(world.tasks),
            resolver=lambda task: (f"fake/{task}", None, dict(split_map)),
        )
    }


class _Recorder:
    """Captures what ``start_run`` / the run logger would persist."""

    def __init__(self, metadata: dict[str, Any]):
        self.metadata = deepcopy(metadata)
        self.events: list[dict[str, Any]] = []
        self.summary: dict[str, Any] | None = None
        self.status: str | None = None
        self.error: Any = None

    def log_event(self, event_type, metrics=None, *, step=None, context=None):
        self.events.append(deepcopy({"event_type": event_type, "metrics": metrics, "step": step, "context": context}))

    def log_summary(self, summary_dict):
        self.summary = deepcopy(summary_dict)

    def finish(self, status, error=None):
        self.status = status
        self.error = error


class _ReachedModelBuild(Exception):
    """Raised by the sentinel classifier build: a config error must fire before this."""


def _install_fakes(monkeypatch, world: World, recorders: list[_Recorder], *, sentinel_build: bool = False) -> None:
    class OpenClipClassifier(_FakeClassifier):
        @staticmethod
        def build(cfg):
            if sentinel_build:
                raise _ReachedModelBuild(cfg.pretrained)
            return world.build_classifier(cfg)

    def build_vision_loaders(*, preprocess, hf_ds, batch_size, **_kwargs):
        tag = _tag_of(preprocess)
        task = hf_ds.task

        def loader(split):
            return DataLoader(_split_dataset(task, split, tag), batch_size=int(batch_size), shuffle=False)

        return SimpleNamespace(
            train=loader("train"), val=loader("val"), test=loader("test"), classnames=_classnames(task)
        )

    def build_vision_calibration_loader(dataset_spec, *, preprocess, batch_size, **_kwargs):
        name = dataset_spec if isinstance(dataset_spec, str) else str(dict(dataset_spec).get("path", dataset_spec))
        return DataLoader(
            _split_dataset(name, "calibration", _tag_of(preprocess), n=16), batch_size=int(batch_size), shuffle=False
        )

    def load_hf_splits(path, config=None, requested_splits=()):
        return SimpleNamespace(task=str(path).split("/")[-1], path=path)

    def extract_classnames(hf_ds, label_key="label", strict=False):
        return _classnames(hf_ds.task)

    def eval_task_top1(*, clf, loaders, classnames, build_cfg_task, device, split, text_features=None):
        eval_loader = loaders.val if split == "val" else loaders.test
        clf.build_zeroshot_text_features(list(classnames), build_cfg_task)
        return float(clf.top1(eval_loader, device=device))

    def start_run(*, entrypoint, logging_cfg, metadata, summary_path=None):
        recorder = _Recorder(metadata)
        recorders.append(recorder)
        return recorder

    fakes: dict[str, Any] = {
        "OpenClipClassifier": OpenClipClassifier,
        "load_ckpt": lambda path, *a, **k: world.checkpoint(str(path)),
        "resolve_ckpt_path": lambda path, *a, **k: str(path),
        "SUITES": _fake_suite(world),
        "load_hf_splits": load_hf_splits,
        "extract_classnames": extract_classnames,
        "build_vision_loaders": build_vision_loaders,
        "build_vision_calibration_loader": build_vision_calibration_loader,
        "eval_task_top1": eval_task_top1,
        "start_run": start_run,
    }
    for module_name in PATCH_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for name, fake in fakes.items():
            monkeypatch.setattr(module, name, fake, raising=False)


# --------------------------------------------------------------------------------------
# Running main()
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    summary: dict[str, Any]
    resolved_config: dict[str, Any]
    events: list[dict[str, Any]]
    pt_files: dict[str, str]  # artifact path relative to the run dir -> hash of its tensors / JSON content
    root: Path


def _base_cfg(world: World, root: Path, **overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "suite": "fake2",
        "tasks": ",".join(world.tasks),
        "device": "cpu",
        "seed": 42,
        "batch_size": 4,
        "num_workers": 0,
        "source_clip_model": "fake-src",
        "source_clip_pretrained": SRC_TAG,
        "target_clip_model": "fake-tgt",
        "target_clip_pretrained": TGT_TAG,
        "tuned_ckpts": world.tuned_ckpts(),
        "method": "theseus",
        "method_params": {"seq_align": "mean", "num_batches": 2, "verbose": False, "show_progress": False},
        "block_extension_params": {"n_batches_act": 2, "verbose": False, "show_progress": False},
        "alpha": 1.0,
        "logging": {"local_log_dir": str(root / "logs")},
    }
    cfg.update(overrides)
    return cfg


def _tree_hash(root: Path) -> dict[str, str]:
    """Hash every artifact main() wrote under the run dir: ``.pt`` tensor dicts and JSON sidecars."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".pt":
            out[rel] = hash_tensor_dict(torch.load(path, map_location="cpu", weights_only=True))
        elif path.suffix == ".json" and rel != "cfg.json" and not rel.startswith("logs/"):
            out[rel] = hash_json(json.loads(path.read_text()))
    return out


def _launch(cfg: dict[str, Any], root: Path, monkeypatch, world: World, *, sentinel_build: bool = False):
    """Install the fakes, write ``cfg`` to JSON, set ``sys.argv`` and call the real ``main()``."""
    from merge_and_rebase.eval import vision_rebase

    recorders: list[_Recorder] = []
    _install_fakes(monkeypatch, world, recorders, sentinel_build=sentinel_build)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "cfg.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["vision_rebase", "--config", str(config_path)])
    try:
        vision_rebase.main()
    except BaseException:
        _launch.last_recorders = recorders  # type: ignore[attr-defined]
        raise
    return recorders


def run_main(cfg: dict[str, Any], root: Path, monkeypatch, *, world: World | None = None) -> RunResult:
    """Run ``vision_rebase.main()`` on the fake world and collect what it produced."""
    world = world or World()
    recorders = _launch(cfg, root, monkeypatch, world)
    assert len(recorders) == 1
    rec = recorders[0]
    assert rec.status == "success" and rec.summary is not None
    return RunResult(
        summary=rec.summary,
        resolved_config=rec.metadata["resolved_config"],
        events=rec.events,
        pt_files=_tree_hash(root),
        root=root,
    )


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------

_SEARCH = {"alpha_search": True, "alpha_min": 0.0, "alpha_max": 1.0, "alpha_step": 0.5}
# Balanced (vision8-mix) transport calibration of the single-transport modes needs batches * (bs / n_tasks) samples
# per task; the tiny splits hold 8.
_CAL = {"transport_calibration_batches": 2}
_TP = {"seq_align": "mean", "num_batches": 2, "verbose": False, "show_progress": False}
_BE = {"n_batches_act": 2, "verbose": False, "show_progress": False}
# Ariadne main configuration, as in the golden `test_vision_rebase_run_direct_residual_fit` (_MAIN).
_ARIADNE = {
    "components": ["mlp.c_proj"],
    "component_target": "block_boundary",
    "activation_storage": "streaming",
    "ridge_estimator": "empirical_bayes",
    "alignment_map": "polar",
    "procrustes_source": "activation",
    "num_batches": 3,
    "ridge_relative": 0.05,
}

_EQ = {"source": (2, 4), "target": (2, 6)}
_SAME = {"source": (2, 4), "target": (2, 4)}
_EXT = {"source": (2, 4), "target": (3, 6)}
_SHR = {"source": (3, 4), "target": (2, 6)}
_DOUBLE = {"source": (2, 4), "target": (4, 6)}  # target-informed protocols address doubled positions


@dataclass(frozen=True)
class Case:
    world: dict[str, Any]
    cfg: dict[str, Any]
    save_tvs: bool = False  # sets save_transported_tvs_dir under the run dir
    save_merged: bool = False  # sets save_merged under the run dir


CASES: dict[str, Case] = {
    # 1: THESEUS, equal depth, fixed alpha, transported TVs saved (native + legacy variants)
    "theseus_equal_depth_none_fixed_alpha_save_tvs": Case(
        _EQ, {"merge_mode": "none", "save_transported_tvs_legacy": True}, save_tvs=True
    ),
    "theseus_samearch_none_alpha_search_untransported": Case(_SAME, {"merge_mode": "none", **_SEARCH}),
    # 2 / 3: THESEUS with a depth mismatch -> BRACE prestep
    "theseus_extend_brace_correction_search_shared": Case(
        _EXT,
        {**_SEARCH, "alpha_selection": "shared", "block_extension_params": {**_BE, "skip_correction": False}},
    ),
    "theseus_extend_brace_skip_correction": Case(
        _EXT, {"alpha": 1.0, "block_extension_params": {**_BE, "skip_correction": True}}
    ),
    # 4 / 5: BiCo with a depth mismatch
    "bico_extend_discrete_index_match": Case(_EXT, {"method": "bico", "depth_alignment": "discrete_index_match"}),
    "bico_extend_depth_alignment_absent": Case(_EXT, {"method": "bico"}),
    # 6: THESEUS shrink
    "theseus_shrink_brace": Case(_SHR, {}),
    # 7: Ariadne (both registry spellings) plus the once-only merged fit
    "ariadne_spelling_ariadne": Case(
        _EXT, {"method": "ariadne", "method_params": {}, "ariadne_params": _ARIADNE}, save_tvs=True
    ),
    "ariadne_spelling_direct_residual": Case(
        _EXT, {"method": "direct_residual", "method_params": {}, "direct_residual_params": _ARIADNE}
    ),
    "ariadne_merge_in_source_then_fit": Case(
        _EXT,
        {
            "method": "ariadne",
            "method_params": {},
            "ariadne_params": {**_ARIADNE, "merge_mode": "merge_in_source_then_fit"},
        },
    ),
    "ariadne_calibration_tiny_imagenet": Case(
        _EXT,
        {"method": "ariadne", "method_params": {}, "ariadne_params": {**_ARIADNE, "calibration_data": "tiny_imagenet"}},
    ),
    "ariadne_calibration_vision8_mix": Case(
        _EXT,
        {"method": "ariadne", "method_params": {}, "ariadne_params": {**_ARIADNE, "calibration_data": "vision8_mix"}},
    ),
    # sequential endpoint construction (write-once vectors + JSON sidecars); the load leg is a separate test.
    # Spelled "direct_residual": the loader only finds files saved under that spelling (see the quirk test).
    "direct_residual_sequential_endpoints_save": Case(
        _EXT,
        {
            "method": "direct_residual",
            "method_params": {},
            "direct_residual_params": {
                "components": ["mlp.c_proj"],
                "endpoint_construction": "sequential_source_endpoints",
                "num_batches": 3,
            },
        },
        save_tvs=True,
    ),
    # 8: rebase_then_merge, shared and hierarchical per_task alpha
    "theseus_rebase_then_merge_shared": Case(_EQ, {"merge_mode": "rebase_then_merge", **_SEARCH}),
    "theseus_rebase_then_merge_per_task_hierarchical": Case(
        _EQ, {"merge_mode": "rebase_then_merge", "alpha_selection": "per_task", **_SEARCH}
    ),
    "theseus_rebase_then_merge_per_task_no_global_search": Case(
        _EQ,
        {"merge_mode": "rebase_then_merge", "alpha_selection": "per_task", "global_alpha_search": False, **_SEARCH},
    ),
    # 9: single-transport merge modes
    "theseus_merge_then_rebase": Case(_EQ, {"merge_mode": "merge_then_rebase", **_CAL, **_SEARCH}),
    "theseus_merge_then_rebase_tiny_protocol": Case(
        _EQ, {"merge_mode": "merge_then_rebase", "transport_calibration_protocol": "tiny", **_SEARCH}
    ),
    "theseus_brace_merge_then_transport": Case(_EXT, {"merge_mode": "brace_merge_then_transport", **_CAL, **_SEARCH}),
    "theseus_merge_then_brace_then_transport_skip_correction": Case(
        _EXT,
        {
            "merge_mode": "merge_then_brace_then_transport",
            **_CAL,
            **_SEARCH,
            "block_extension_params": {**_BE, "skip_correction": True},
        },
    ),
    "theseus_merge_then_brace_then_transport_correction": Case(
        _EXT,
        {
            "merge_mode": "merge_then_brace_then_transport",
            **_CAL,
            **_SEARCH,
            "block_extension_params": {
                **_BE,
                "calibration_dataset": {"path": "zh-plus/tiny-imagenet", "split": "valid"},
            },
        },
    ),
    # 10: native target tasks
    "theseus_native_target_explicit": Case(
        {**_EXT, "native": ("DTD",)},
        {"merge_mode": "rebase_then_merge", "native_target_tasks": ["DTD"], **_SEARCH},
    ),
    "theseus_native_target_auto_detected_per_task": Case(
        {**_EXT, "native": ("DTD",)},
        {"merge_mode": "brace_transport_then_merge", "alpha_selection": "per_task", **_SEARCH},
    ),
    # 11: same-depth direct_target completion (transport-free)
    "theseus_same_depth_direct_target": Case(
        _EQ,
        {
            "block_extension_params": {
                **_BE,
                "skip_correction": True,
                "target_residual_completion": {
                    "enabled": True,
                    "mode": "direct_target",
                    "target_scope": "all",
                    "num_batches": 2,
                },
            }
        },
    ),
    # 12: alpha early stopping
    "theseus_alpha_patience_shared": Case(
        _EQ,
        {
            "merge_mode": "rebase_then_merge",
            "alpha_patience": 0,
            "alpha_min": 0.0,
            "alpha_max": 2.0,
            "alpha_step": 0.25,
            "alpha_search": True,
        },
    ),
    "theseus_alpha_patience_per_task": Case(
        _EQ,
        {
            "alpha_selection": "per_task",
            "alpha_patience": 1,
            "alpha_min": 0.0,
            "alpha_max": 2.0,
            "alpha_step": 0.25,
            "alpha_search": True,
        },
    ),
    # 13: save_merged
    "theseus_rebase_then_merge_save_merged": Case(
        _EQ, {"merge_mode": "rebase_then_merge", **_SEARCH}, save_merged=True
    ),
    # 14: independent endpoint average base construction
    "theseus_independent_endpoint_average": Case(
        _EXT,
        {"merge_mode": "brace_transport_then_merge", "base_construction": "independent_endpoint_average", **_SEARCH},
    ),
    # eval_before_rebase / source LMC around the BRACE prestep
    "theseus_extend_eval_before_and_source_lmc": Case(
        _EXT,
        {
            "eval_before_rebase": True,
            "source_lmc_eval": True,
            "source_lmc_alpha_step": 0.5,
            "cross_task_lmc_pairs": [["MNIST", "DTD"]],
            "all_task_lmc_tasks": ["MNIST", "DTD"],
        },
    ),
    "theseus_samedepth_eval_before_rebase": Case(_EQ, {"eval_before_rebase": True}),
    # target-informed BRACE protocols (all need a depth-mismatched pair + lmc_mode=shared)
    "theseus_double_target_residual_completion": Case(
        _DOUBLE,
        {
            "block_extension_params": {
                **_BE,
                "lmc_mode": "shared",
                "target_residual_completion": {"enabled": True, "num_batches": 2},
            }
        },
    ),
    "theseus_double_target_residual_completion_direct_target": Case(
        _DOUBLE,
        {
            "block_extension_params": {
                **_BE,
                "skip_correction": True,
                "target_residual_completion": {"enabled": True, "mode": "direct_target", "num_batches": 2},
            }
        },
    ),
    "theseus_double_joint_blockwise_correction": Case(
        _DOUBLE,
        {
            "block_extension_params": {
                **_BE,
                "lmc_mode": "shared",
                "extension_strategy": "duplicate_per_weight",
                "calibration_split": "val",
                "joint_blockwise_correction": {"enabled": True},
            }
        },
    ),
    "theseus_double_direct_p1_correction": Case(
        _DOUBLE,
        {
            "block_extension_params": {
                **_BE,
                "lmc_mode": "shared",
                "extension_strategy": "duplicate_per_weight",
                "calibration_split": "val",
                "direct_p1_correction": {"enabled": True},
            }
        },
    ),
    # task-independent transport calibration
    "theseus_transport_calibration_tiny_imagenet": Case(_EQ, {"transport_calibration_data": "tiny_imagenet"}),
    # 15: other rebase methods
    "gradfix_same_architecture": Case(
        _SAME,
        {
            "method": "gradfix",
            "method_params": {},
            "grad_batch_size": 4,
            "grad_imgs_per_class": 2,
            "grad_num_batches": 2,
        },
    ),
    # transfusion is not driven: `_load_or_compute_permutations` raises 'requires CUDA' on CPU.
}

# P5.12: depth-mismatched THESEUS without an explicit skip_correction (and BiCo without depth_alignment) would
# change meaning under the per-method depth defaults; these pins stay on the legacy semantics explicitly.
_LEGACY_DEPTH_CASES = (
    "bico_extend_depth_alignment_absent",
    "theseus_brace_merge_then_transport",
    "theseus_double_direct_p1_correction",
    "theseus_double_joint_blockwise_correction",
    "theseus_double_target_residual_completion",
    "theseus_extend_eval_before_and_source_lmc",
    "theseus_independent_endpoint_average",
    "theseus_merge_then_brace_then_transport_correction",
    "theseus_native_target_auto_detected_per_task",
    "theseus_native_target_explicit",
    "theseus_shrink_brace",
)
for _name in _LEGACY_DEPTH_CASES:
    CASES[_name] = replace(CASES[_name], cfg={**CASES[_name].cfg, "depth_defaults": "legacy"})
# New per-method defaults (depth_defaults="method"): THESEUS -> skip_correction=True, BiCo -> discrete_index_match.
CASES["theseus_extend_depth_defaults_method"] = Case(_EXT, {"alpha": 1.0, "depth_defaults": "method"})
CASES["bico_extend_depth_defaults_method"] = Case(_EXT, {"method": "bico", "depth_defaults": "method"})
CASES["theseus_shrink_depth_defaults_method"] = Case(_SHR, {"depth_defaults": "method"})


# --------------------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------------------


def _normalize(obj: Any, root: Path) -> Any:
    """Replace the run directory by a placeholder so path-valued fields compare across runs."""
    if isinstance(obj, str):
        return obj.replace(str(root), "<RUN>")
    if isinstance(obj, Mapping):
        return {_normalize(k, root) if isinstance(k, str) else k: _normalize(v, root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v, root) for v in obj]
    return obj


def _key_paths(obj: Any, prefix: str = "") -> set[str]:
    """Every key path of a nested summary: ``a.b`` for mapping keys, ``a[]`` for list elements."""
    paths: set[str] = set()
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.add(child)
            paths |= _key_paths(value, child)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            paths |= _key_paths(value, f"{prefix}[]")
    return paths


def _digest(result: RunResult) -> dict[str, str]:
    out = {
        "summary": hash_json(_normalize(result.summary, result.root)),
        "resolved_config": hash_json(_normalize(result.resolved_config, result.root)),
        "events": hash_json(_normalize(result.events, result.root)),
    }
    for name, digest in sorted(result.pt_files.items()):
        out[f"file:{name}"] = digest
    return out


def _structure(result: RunResult) -> dict[str, list[str]]:
    return {
        "summary_keys": sorted(_key_paths(_normalize(result.summary, result.root))),
        "event_names": [e["event_type"] for e in result.events],
        "saved_files": sorted(result.pt_files),
    }


def _capture(lines: dict[str, str], structure: dict[str, Any] | None = None, case: str | None = None) -> bool:
    capture = os.environ.get("GOLDEN_CAPTURE")
    if not capture:
        return False
    with open(capture, "a") as fh:
        for name, digest in lines.items():
            fh.write(f"{name} {digest}\n")
    target = os.environ.get("GOLDEN_CAPTURE_STRUCTURE")
    if target and structure is not None and case is not None:
        path = Path(target)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[case] = structure
        path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return True


def _load_structure() -> dict[str, Any]:
    return json.loads(STRUCTURE_PATH.read_text()) if STRUCTURE_PATH.exists() else {}


def _run_case(name: str, root: Path, monkeypatch, *, world: World | None = None, **cfg_updates: Any) -> RunResult:
    case = CASES[name]
    world = world or World(**case.world)
    overrides = dict(case.cfg)
    overrides.update(cfg_updates)
    if case.save_tvs:
        overrides["save_transported_tvs_dir"] = str(root / "tvs")
    if case.save_merged:
        overrides["save_merged"] = str(root / "merged" / "merged.pt")
    return run_main(_base_cfg(world, root, **overrides), root, monkeypatch, world=world)


# --------------------------------------------------------------------------------------
# End-to-end pins
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_main_golden(case, tmp_path, monkeypatch):
    first = _run_case(case, tmp_path / "run1", monkeypatch)
    second = _run_case(case, tmp_path / "run2", monkeypatch)
    digest = _digest(first)
    # Two runs in one process: every pinned artifact must be bit-stable.
    assert _digest(second) == digest, f"{case}: main() is not deterministic within one process"
    assert _structure(second) == _structure(first)
    # Weight-sensitivity guard: a pin over an empty / constant summary would pin nothing.
    assert first.summary["test_results"]["per_task_absolute_accuracy"]

    structure = _structure(first)
    lines = {f"{case}:{part}": value for part, value in digest.items()}
    if _capture(lines, structure, case):
        return
    expected_structure = _load_structure().get(case)
    assert expected_structure is not None, f"no recorded structure for {case!r}"
    for part in ("summary_keys", "event_names", "saved_files"):
        assert structure[part] == expected_structure[part], (
            f"{case}: {part} changed\n"
            f"  added  : {sorted(set(structure[part]) - set(expected_structure[part]))}\n"
            f"  removed: {sorted(set(expected_structure[part]) - set(structure[part]))}"
        )
    expected = {k[len(case) + 1 :]: v for k, v in EXPECTED.items() if k.startswith(case + ":")}
    assert expected, f"no expected hashes recorded for {case!r} (actual {digest})"
    assert digest == expected, (
        f"{case}: golden hashes changed\n  expected {json.dumps(expected, indent=1, sort_keys=True)}\n"
        f"  actual   {json.dumps(digest, indent=1, sort_keys=True)}"
    )


def _load_leg_cfg(root: Path) -> dict[str, Any]:
    """The ``direct_residual_sequential_endpoints_save`` config, loading the vectors that run saved instead."""
    case = CASES["direct_residual_sequential_endpoints_save"]
    overrides = dict(case.cfg)
    overrides["load_direct_residual_tvs_dir"] = str(root.parent / "run1" / "tvs")
    return _base_cfg(World(**case.world), root, **overrides)


def test_main_golden_sequential_load(tmp_path, monkeypatch):
    """Save leg (a CASES entry) then load leg: the loader re-verifies fit provenance and the tensor hash."""
    _run_case("direct_residual_sequential_endpoints_save", tmp_path / "run1", monkeypatch)
    first = run_main(_load_leg_cfg(tmp_path / "run2"), tmp_path / "run2", monkeypatch, world=World(**_EXT))
    again = run_main(_load_leg_cfg(tmp_path / "run3"), tmp_path / "run3", monkeypatch, world=World(**_EXT))
    digest = _digest(first)
    assert _digest(again) == digest
    assert first.summary["direct_residual"]["loaded_vectors_by_task"]
    case = "direct_residual_sequential_endpoints_load"
    structure = _structure(first)
    if _capture({f"{case}:{k}": v for k, v in digest.items()}, structure, case):
        return
    expected_structure = _load_structure()[case]
    assert structure == expected_structure
    assert digest == {k[len(case) + 1 :]: v for k, v in EXPECTED.items() if k.startswith(case + ":")}


def test_main_source_only_completes_without_transport(tmp_path, monkeypatch):
    """B2 (fixed): source_only used to crash on a zip length mismatch; it now finishes with the source-side rows."""
    world = World(**_EQ)
    result = run_main(_base_cfg(world, tmp_path, source_only=True), tmp_path, monkeypatch, world=world)
    assert result.summary["source_only"] is True
    assert result.summary["target_hash_before"] == result.summary["target_hash_after"]
    assert result.pt_files == {}


def test_main_sequential_dr_refuses_overwrite(tmp_path, monkeypatch):
    """Sequential Ariadne vectors are write-once: a second save into the same directory must fail loudly."""
    _run_case("direct_residual_sequential_endpoints_save", tmp_path / "run1", monkeypatch)
    case = CASES["direct_residual_sequential_endpoints_save"]
    cfg = _base_cfg(
        World(**case.world),
        tmp_path / "run2",
        **{**case.cfg, "save_transported_tvs_dir": str(tmp_path / "run1" / "tvs")},
    )
    with pytest.raises(
        FileExistsError, match=r"refusing to overwrite sequential DR vector: .*_transported_native\.pt$"
    ):
        run_main(cfg, tmp_path / "run2", monkeypatch, world=World(**case.world))


# --------------------------------------------------------------------------------------
# Error surface: invalid configs, the exact message, and whether they fail BEFORE the
# classifiers are built (sentinel build) and before / after the run record is opened.
# --------------------------------------------------------------------------------------


def _drop_block0_c_proj(sd):
    sd.pop("visual.transformer.resblocks.0.mlp.c_proj.weight")


def _prefix_junk(sd):
    for key in list(sd):
        sd["junk." + key] = sd.pop(key)


def E(message: str) -> str:
    return re.escape(message)


def P(prefix: str) -> str:
    """Prefix match, for messages whose tail is long (the head names the failing check)."""
    return re.escape(prefix) + ".*"


@dataclass(frozen=True)
class Err:
    cfg: dict[str, Any]
    exc: type[BaseException]
    message: str  # regex, re.fullmatch with DOTALL
    before_build: bool = True  # raised before OpenClipClassifier.build
    run_started: bool = False  # start_run() already called (a run record exists, finished "failed")
    world: dict[str, Any] = field(default_factory=lambda: dict(_EQ))
    mutate: tuple[str, Any] | None = None  # (ckpt path, fn(state_dict))


_BE_SKIP = {**_BE, "skip_correction": True}
_JOINT = {**_BE, "lmc_mode": "shared", "extension_strategy": "duplicate_per_weight", "calibration_split": "val"}
_NATIVE_DTD = {**_EQ, "native": ("DTD",)}
_ARIADNE_ONLY = {"method": "ariadne", "method_params": {}}

ERRORS: dict[str, Err] = {
    # ---- before any model is built, no run record ------------------------------------------------
    "method_params_not_dict": Err(
        {"method_params": [1]}, ValueError, E("config['method_params'] must be a dict when provided.")
    ),
    "ariadne_and_direct_residual_params": Err(
        {**_ARIADNE_ONLY, "ariadne_params": {}, "direct_residual_params": {}},
        ValueError,
        E("config has both 'direct_residual_params' and its alias 'ariadne_params'; use one"),
    ),
    "load_dr_tvs_requires_sequential": Err(
        {**_ARIADNE_ONLY, "load_direct_residual_tvs_dir": "/nowhere"},
        ValueError,
        E("load_direct_residual_tvs_dir requires a sequential endpoint construction"),
    ),
    "load_dr_tvs_and_save": Err(
        {
            **_ARIADNE_ONLY,
            "ariadne_params": {"components": ["mlp.c_proj"], "endpoint_construction": "sequential_source_endpoints"},
            "load_direct_residual_tvs_dir": "/nowhere",
            "save_transported_tvs_dir": "/elsewhere",
        },
        ValueError,
        E("cannot save transported artifacts while loading sequential DR vectors"),
    ),
    "ariadne_unknown_param": Err(
        {**_ARIADNE_ONLY, "ariadne_params": {"bogus": 1}}, ValueError, E("unknown direct_residual fields: ['bogus']")
    ),
    "ariadne_unknown_preset": Err(
        {**_ARIADNE_ONLY, "ariadne_params": {"preset": "nope"}},
        ValueError,
        E("unknown direct_residual preset 'nope'; valid presets: ['ariadne']"),
    ),
    "depth_alignment_unknown": Err(
        {"depth_alignment": "nope"}, ValueError, E("depth_alignment must be one of: ariadne, discrete_index_match")
    ),
    "ariadne_merge_in_source_per_task_alpha": Err(
        {**_ARIADNE_ONLY, "ariadne_params": {"merge_mode": "merge_in_source_then_fit"}, "alpha_selection": "per_task"},
        ValueError,
        P("direct_residual merge_mode='merge_in_source_then_fit' requires alpha_selection='shared': "),
    ),
    "block_extension_eval_split": Err(
        {"block_extension_eval_split": "train"}, ValueError, E("block_extension_eval_split must be one of: val, test")
    ),
    "source_lmc_eval_split": Err(
        {"source_lmc_eval_split": "train"}, ValueError, E("source_lmc_eval_split must be one of: val, test")
    ),
    "source_lmc_alpha_step_zero": Err({"source_lmc_alpha_step": 0}, ValueError, E("source_lmc_alpha_step must be > 0")),
    "cross_task_pairs_not_a_list": Err(
        {"cross_task_lmc_pairs": "MNIST"}, ValueError, E("cross_task_lmc_pairs must be a list of two-task lists.")
    ),
    "cross_task_pair_wrong_length": Err(
        {"cross_task_lmc_pairs": [["MNIST"]]},
        ValueError,
        E("Each cross_task_lmc_pairs item must contain exactly two task names."),
    ),
    "cross_task_pair_same_task": Err(
        {"cross_task_lmc_pairs": [["MNIST", "MNIST"]]},
        ValueError,
        E("cross_task_lmc_pairs cannot interpolate a task with itself."),
    ),
    "cross_task_eval_split": Err(
        {"cross_task_lmc_eval_split": "x"}, ValueError, E("cross_task_lmc_eval_split must be one of: val, test")
    ),
    "all_task_lmc_not_a_list": Err(
        {"all_task_lmc_tasks": "MNIST"}, ValueError, E("all_task_lmc_tasks must be a list of task names.")
    ),
    "all_task_lmc_single_task": Err(
        {"all_task_lmc_tasks": ["MNIST"]},
        ValueError,
        E("all_task_lmc_tasks must contain at least two distinct task names."),
    ),
    "all_task_lmc_duplicates": Err(
        {"all_task_lmc_tasks": ["MNIST", "MNIST"]},
        ValueError,
        E("all_task_lmc_tasks must contain at least two distinct task names."),
    ),
    "all_task_eval_split": Err(
        {"all_task_lmc_eval_split": "x"}, ValueError, E("all_task_lmc_eval_split must be one of: val, test")
    ),
    "alpha_patience_negative": Err({"alpha_patience": -1}, ValueError, E("alpha_patience must be >= 0")),
    "alpha_search_split": Err(
        {"alpha_search_split": "train"}, ValueError, E("alpha_search_split must be one of: val, test")
    ),
    "alpha_step_zero": Err({"alpha_search": True, "alpha_step": 0}, ValueError, E("alpha_step must be > 0")),
    "alpha_max_below_min": Err(
        {"alpha_search": True, "alpha_min": 1.0, "alpha_max": 0.5}, ValueError, E("alpha_max must be >= alpha_min")
    ),
    "alpha_selection_unknown": Err(
        {"alpha_selection": "x"}, ValueError, E("alpha_selection must be one of: shared, per_task")
    ),
    "merge_mode_unknown": Err(
        {"merge_mode": "x"},
        ValueError,
        E(
            "merge_mode must be one of: none, rebase_then_merge, merge_then_rebase, brace_transport_then_merge, "
            "brace_merge_then_transport, merge_then_brace_then_transport"
        ),
    ),
    "single_transport_mode_per_task_alpha": Err(
        {"merge_mode": "merge_then_rebase", "alpha_selection": "per_task"},
        ValueError,
        P("merge_then_rebase requires alpha_selection='shared': per-task alpha search is only defined for "),
    ),
    "merge_method_unknown": Err(
        {"merge_mode": "rebase_then_merge", "merge_method": "x"},
        KeyError,
        r"\"Unknown merge method 'x'\. Available: .*",
    ),
    "merge_params_not_a_mapping": Err(
        {"merge_mode": "rebase_then_merge", "merge_params": [1]},
        ValueError,
        E("merge_params must be a JSON object / mapping when provided."),
    ),
    "global_alpha_search_not_bool": Err(
        {"merge_mode": "rebase_then_merge", "global_alpha_search": "yes"},
        ValueError,
        E("global_alpha_search must be a boolean (true/false)."),
    ),
    "ariadne_with_single_transport_merge_mode": Err(
        {**_ARIADNE_ONLY, "merge_mode": "merge_then_rebase"},
        ValueError,
        P("method='direct_residual' does not support merge_mode='merge_then_rebase': "),
    ),
    "base_construction_unknown": Err(
        {"base_construction": "x"},
        ValueError,
        E("base_construction must be one of: per_task, independent_endpoint_average"),
    ),
    "alpha_search_without_positive_alpha": Err(
        {"alpha_search": True, "alpha_min": 0.0, "alpha_max": 0.0},
        ValueError,
        E("alpha_search requires at least one alpha > 0."),
    ),
    "suite_unknown": Err({"suite": "x"}, ValueError, E("Unknown suite 'x'. Available: ['fake2']")),
    "tasks_unknown": Err(
        {"tasks": "MNIST,Bogus"}, ValueError, E("Unknown tasks: ['Bogus']. Allowed: ['DTD', 'MNIST']")
    ),
    "block_extension_misplaced_top_level_key": Err(
        {"target_residual_completion": {"enabled": True}},
        ValueError,
        P("Found ['target_residual_completion'] at the top level of the run config; "),
    ),
    "block_extension_params_not_a_mapping": Err(
        {"block_extension_params": [1]}, ValueError, E("config['block_extension_params'] must be a dict when provided.")
    ),
    "block_extension_n_batches_act_zero": Err(
        {"block_extension_params": {"n_batches_act": 0}},
        ValueError,
        E("block_extension_params.n_batches_act must be > 0."),
    ),
    "block_extension_ridge_identity_negative": Err(
        {"block_extension_params": {"ridge_identity": -1}},
        ValueError,
        E("block_extension_params.ridge_identity must be >= 0."),
    ),
    "block_extension_identity_block_needs_skip_correction": Err(
        {"block_extension_params": {"inserted_block_mode": "residual_identity"}},
        ValueError,
        P("block_extension_params.inserted_block_mode='residual_identity' requires skip_correction=true: "),
    ),
    "block_extension_joint_needs_shared_lmc": Err(
        {"block_extension_params": {"joint_blockwise_correction": {"enabled": True}}},
        ValueError,
        E("joint_blockwise_correction requires lmc_mode='shared'."),
    ),
    "block_extension_two_target_informed_protocols": Err(
        {
            "block_extension_params": {
                "lmc_mode": "shared",
                "target_residual_completion": {"enabled": True},
                "direct_p1_correction": {"enabled": True},
            }
        },
        ValueError,
        P("target_residual_completion, joint_blockwise_correction, and direct_p1_correction are mutually exclusive; "),
    ),
    # ---- after start_run (a run record exists) but still before any model is built ---------------
    "tuned_ckpts_missing": Err(
        {"tuned_ckpts": None},
        ValueError,
        E("Provide tuned checkpoints via --tuned-ckpts or config 'tuned_ckpts'."),
        run_started=True,
    ),
    # ---- after the classifiers are built ------------------------------------------------------------
    "native_target_task_not_in_task_list": Err(
        {"native_target_tasks": ["Bogus"]},
        ValueError,
        E("native_target_tasks contains tasks not in the task list: ['Bogus']"),
        before_build=False,
        run_started=True,
    ),
    "shrink_direct_target_needs_scope_all": Err(
        {
            "block_extension_params": {
                **_BE_SKIP,
                "target_residual_completion": {"enabled": True, "mode": "direct_target"},
            }
        },
        ValueError,
        P("Shrink direct_target completion requires target_scope='all': "),
        before_build=False,
        run_started=True,
        world=dict(_SHR),
    ),
    "discrete_index_match_with_target_informed_protocol": Err(
        {
            "depth_alignment": "discrete_index_match",
            "block_extension_params": {
                **_BE_SKIP,
                "target_residual_completion": {"enabled": True, "mode": "direct_target", "target_scope": "all"},
            },
        },
        ValueError,
        P("depth_alignment='discrete_index_match' is incompatible with target_residual_completion, "),
        before_build=False,
        run_started=True,
        world=dict(_DOUBLE),
    ),
    "joint_correction_needs_theseus_or_bico": Err(
        {
            "method": "gradfix",
            "method_params": {},
            "block_extension_params": {**_JOINT, "joint_blockwise_correction": {"enabled": True}},
        },
        ValueError,
        E("Joint/direct P1 correction requires a Theseus- or BiCo-like transport method"),
        before_build=False,
        run_started=True,
        world=dict(_SAME),
    ),
    "joint_correction_needs_depth_mismatch": Err(
        {"block_extension_params": {**_JOINT, "joint_blockwise_correction": {"enabled": True}}},
        ValueError,
        P("Joint/direct P1 correction requires a depth-mismatched source/target pair "),
        before_build=False,
        run_started=True,
        world=dict(_SAME),
    ),
    "merge_then_rebase_with_block_extension_prestep": Err(
        {"merge_mode": "merge_then_rebase"},
        NotImplementedError,
        P("merge_then_rebase does not support the block-extension prestep yet: "),
        before_build=False,
        run_started=True,
        world=dict(_EXT),
    ),
    "attn_patch_cfg_not_a_dict": Err(
        {"attn_patch_cfg": [1]},
        ValueError,
        E("config['attn_patch_cfg'] must be a dict when provided."),
        before_build=False,
        run_started=True,
    ),
    "checkpoint_matches_neither_base": Err(
        {},
        ValueError,
        P(
            "Tuned checkpoint for task 'MNIST' matches neither the source nor the target visual backbone (ckpt://MNIST). "
        ),
        before_build=False,
        run_started=True,
        mutate=("ckpt://MNIST", _prefix_junk),
    ),
    "target_architecture_checkpoint_without_auto_detect": Err(
        {"native_target_tasks": ["MNIST"], "auto_detect_ckpt_base": False, "merge_mode": "rebase_then_merge"},
        ValueError,
        P("Tuned checkpoint for task 'DTD' matches the target architecture; add it to native_target_tasks or set "),
        before_build=False,
        run_started=True,
        world=dict(_NATIVE_DTD),
    ),
    # B3 (fixed in P5.14): used to complete with an empty task vector.
    "target_architecture_checkpoint_without_auto_detect_or_native_list": Err(
        {"auto_detect_ckpt_base": False, "merge_mode": "rebase_then_merge"},
        ValueError,
        P("Tuned checkpoint for task 'DTD' matches the target architecture; add it to native_target_tasks or set "),
        before_build=False,
        run_started=True,
        world=dict(_NATIVE_DTD),
    ),
    "strict_load_partial_visual_coverage": Err(
        {"strict_load": True},
        ValueError,
        E("Strict visual checkpoint coverage failed for task 'MNIST': coverage=0.965517, expected=1.0 (ckpt://MNIST)."),
        before_build=False,
        run_started=True,
        mutate=("ckpt://MNIST", _drop_block0_c_proj),
    ),
    "native_target_task_with_merge_mode_none": Err(
        {"native_target_tasks": ["DTD"]},
        ValueError,
        P("Native target checkpoints require a merge mode; merge_mode='none' evaluates "),
        before_build=False,
        run_started=True,
        world=dict(_NATIVE_DTD),
    ),
    "native_target_task_with_single_transport_mode": Err(
        {"native_target_tasks": ["DTD"], "merge_mode": "merge_then_rebase"},
        ValueError,
        P("Native target checkpoints cannot participate in merge_then_rebase: "),
        before_build=False,
        run_started=True,
        world=dict(_NATIVE_DTD),
    ),
    "native_target_task_with_transfusion": Err(
        {
            "method": "transfusion",
            "method_params": {},
            "native_target_tasks": ["DTD"],
            "merge_mode": "rebase_then_merge",
        },
        NotImplementedError,
        P("Native target checkpoints with transfusion are not supported: "),
        before_build=False,
        run_started=True,
        world=dict(_NATIVE_DTD),
    ),
    "independent_endpoint_average_needs_transport_then_merge": Err(
        {"base_construction": "independent_endpoint_average"},
        ValueError,
        E("base_construction='independent_endpoint_average' requires merge_mode='brace_transport_then_merge'."),
        before_build=False,
        run_started=True,
    ),
    "independent_endpoint_average_needs_shared_alpha": Err(
        {
            "base_construction": "independent_endpoint_average",
            "merge_mode": "rebase_then_merge",
            "alpha_selection": "per_task",
        },
        ValueError,
        P("base_construction='independent_endpoint_average' requires alpha_selection='shared'; "),
        before_build=False,
        run_started=True,
    ),
    "independent_endpoint_average_rejects_native_tasks": Err(
        {
            "base_construction": "independent_endpoint_average",
            "merge_mode": "rebase_then_merge",
            "native_target_tasks": ["DTD"],
        },
        ValueError,
        P("base_construction='independent_endpoint_average' requires every task to be an independently transformed "),
        before_build=False,
        run_started=True,
        world=dict(_NATIVE_DTD),
    ),
    "transport_calibration_data_unknown": Err(
        {"transport_calibration_data": "x"},
        ValueError,
        E("transport_calibration_data must be one of ('task_local', 'tiny_imagenet'), got 'x'"),
        before_build=False,
        run_started=True,
    ),
    "transport_calibration_data_needs_theseus_or_bico": Err(
        {"method": "gradfix", "method_params": {}, "transport_calibration_data": "tiny_imagenet"},
        ValueError,
        P("transport_calibration_data applies to THESEUS/BiCo only"),
        before_build=False,
        run_started=True,
        world=dict(_SAME),
    ),
    "save_transported_artifacts_without_dir": Err(
        {"save_transported_artifacts": True},
        ValueError,
        E("save_transported_artifacts=true requires save_transported_tvs_dir."),
        before_build=False,
        run_started=True,
    ),
    "transport_calibration_protocol_unsupported": Err(
        {"merge_mode": "merge_then_rebase", "transport_calibration_protocol": "x"},
        ValueError,
        E("Unsupported transport_calibration_protocol: 'x'"),
        before_build=False,
        run_started=True,
    ),
    "ariadne_vision8_mix_batch_size_not_divisible": Err(
        {**_ARIADNE_ONLY, "ariadne_params": {"calibration_data": "vision8_mix"}, "batch_size": 5},
        ValueError,
        E("calibration_data='vision8_mix' needs batch_size divisible by the 2 calibrated tasks (got batch_size=5)."),
        before_build=False,
        run_started=True,
    ),
    "tuned_checkpoint_aligns_no_tensors": Err(
        {"auto_detect_ckpt_base": False},
        ValueError,
        P("No tensors from tuned checkpoint aligned to source base keys for task 'MNIST': ckpt://MNIST. "),
        before_build=False,
        run_started=True,
        mutate=("ckpt://MNIST", _prefix_junk),
    ),
    # Behaviour pinned as-is (see HASHES.md "Observed quirks"): these are NOT designed validations.
    "weights_length_mismatch_merge_mode_none": Err(
        {"weights": [1.0]},
        ValueError,
        E("zip() argument 2 is shorter than argument 1"),
        before_build=False,
        run_started=True,
    ),
    "weights_length_mismatch_merge": Err(
        {"weights": [1.0], "merge_mode": "rebase_then_merge"},
        ValueError,
        E("weights length must match tuned checkpoints"),
        before_build=False,
        run_started=True,
    ),
}

# P5.12: this error predates the per-method depth defaults; pin it on the legacy semantics explicitly.
for _name in ("merge_then_rebase_with_block_extension_prestep", "joint_correction_needs_depth_mismatch"):
    ERRORS[_name] = replace(ERRORS[_name], cfg={**ERRORS[_name].cfg, "depth_defaults": "legacy"})


def _error_run(name: str, tmp_path: Path, monkeypatch, *, sentinel_build: bool):
    err = ERRORS[name]
    world = World(**err.world)
    if err.mutate is not None:
        world.mutate_ckpt[err.mutate[0]] = err.mutate[1]
    cfg = _base_cfg(world, tmp_path, **err.cfg)
    with pytest.raises(BaseException) as info:
        _launch(cfg, tmp_path, monkeypatch, world, sentinel_build=sentinel_build)
    return info.value, _launch.last_recorders, world  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", sorted(ERRORS))
def test_main_error_surface(name, tmp_path, monkeypatch):
    err = ERRORS[name]
    exc, recorders, _world = _error_run(name, tmp_path / "sentinel", monkeypatch, sentinel_build=True)
    if err.before_build:
        # Fires before OpenClipClassifier.build: the sentinel must NOT have been reached.
        assert not isinstance(exc, _ReachedModelBuild), f"{name}: expected to fail before the model build"
    else:
        # Fires after the build: with the sentinel the run stops AT the build instead.
        assert isinstance(exc, _ReachedModelBuild), f"{name}: expected to reach the model build, got {exc!r}"
        exc, recorders, world = _error_run(name, tmp_path / "real", monkeypatch, sentinel_build=False)
        assert set(world.built) == {SRC_TAG, TGT_TAG}
    assert type(exc) is err.exc, f"{name}: {type(exc).__name__}: {exc}"
    assert re.fullmatch(err.message, str(exc), re.DOTALL), f"{name}: message changed: {str(exc)!r}"
    assert (len(recorders) == 1) is err.run_started
    if err.run_started:
        assert recorders[0].status == "failed" and recorders[0].summary is None
        assert recorders[0].error["type"] == type(exc).__name__


# --------------------------------------------------------------------------------------
# Sanity: the pins are weight-sensitive (a refactor that changed a number cannot hide).
# --------------------------------------------------------------------------------------


def _bump(key: str, amount: float):
    def mutate(sd):
        sd[key] = sd[key] + amount

    return mutate


@pytest.mark.parametrize("name", ["theseus_equal_depth_none_fixed_alpha_save_tvs", "theseus_rebase_then_merge_shared"])
def test_pins_change_when_a_tuned_checkpoint_changes(name, tmp_path, monkeypatch):
    base = _run_case(name, tmp_path / "base", monkeypatch)
    world = World(**CASES[name].world)
    world.mutate_ckpt["ckpt://MNIST"] = _bump("visual.transformer.resblocks.0.mlp.c_proj.weight", 0.01)
    changed = _run_case(name, tmp_path / "changed", monkeypatch, world=world)
    assert _digest(changed)["summary"] != _digest(base)["summary"]
    if CASES[name].save_tvs:
        key = "tvs/MNIST_theseus_transported_native.pt"
        assert changed.pt_files[key] != base.pt_files[key]
        # ... and only the perturbed task's vector moves.
        other = "tvs/DTD_theseus_transported_native.pt"
        assert changed.pt_files[other] == base.pt_files[other]


def test_pins_change_when_one_transported_task_vector_changes(tmp_path, monkeypatch):
    from merge_and_rebase.rebase import get_method

    name = "theseus_equal_depth_none_fixed_alpha_save_tvs"
    base = _run_case(name, tmp_path / "base", monkeypatch)

    method = get_method("theseus")
    original = type(method).transport

    def perturbed(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        out = dict(out)
        key = sorted(out)[-1]
        out[key] = out[key] + 0.05
        return out

    with monkeypatch.context() as local:
        local.setattr(type(method), "transport", perturbed)
        changed = _run_case(name, tmp_path / "changed", monkeypatch)
    assert _digest(changed)["summary"] != _digest(base)["summary"]
    assert (
        changed.pt_files["tvs/DTD_theseus_transported_native.pt"]
        != base.pt_files["tvs/DTD_theseus_transported_native.pt"]
    )


def test_main_sequential_load_reads_vectors_saved_under_the_ariadne_spelling(tmp_path, monkeypatch):
    """Declared change (P5.1b): the loader resolves ``{task}_{ariadne|direct_residual}_...``, so vectors saved by
    ``method='ariadne'`` can be reloaded (previously a FileNotFoundError on the hardcoded ``direct_residual`` name)."""
    case = CASES["direct_residual_sequential_endpoints_save"]
    params = case.cfg["direct_residual_params"]
    ariadne = {"method": "ariadne", "method_params": {}, "ariadne_params": params, "direct_residual_params": None}
    ariadne = {k: v for k, v in ariadne.items() if v is not None}
    save_cfg = _base_cfg(
        World(**case.world), tmp_path / "run1", **ariadne, save_transported_tvs_dir=str(tmp_path / "run1" / "tvs")
    )
    saved = run_main(save_cfg, tmp_path / "run1", monkeypatch, world=World(**case.world))
    assert "tvs/MNIST_ariadne_transported_native.pt" in saved.pt_files
    load_cfg = _base_cfg(
        World(**case.world), tmp_path / "run2", **ariadne, load_direct_residual_tvs_dir=str(tmp_path / "run1" / "tvs")
    )
    loaded = run_main(load_cfg, tmp_path / "run2", monkeypatch, world=World(**case.world))
    by_task = loaded.summary["direct_residual"]["loaded_vectors_by_task"]
    assert by_task and all(str(meta["path"]).endswith("_ariadne_transported_native.pt") for meta in by_task.values())
