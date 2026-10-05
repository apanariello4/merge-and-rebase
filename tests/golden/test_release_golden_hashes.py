"""Phase 0 golden-hash safety net for the release refactor.

Every test below runs a tiny, fully seeded, CPU/single-thread computation through a
CURRENT public module path and compares a SHA-256 of its numerical output
(``_hashing.hash_tensor_dict``, self-contained: raw bytes of every tensor) with a value
captured at commit 7a7ee9a (behaviour-identical to b6128d3). A refactor that changes a
single bit of any pinned output fails loudly here. See ``HASHES.md`` for the table of
cases, what each pins, and the platform caveat (hashes are CPU / 1 thread / this torch build).

Never "fix" a failing hash by pasting the new value: first establish whether the change
is an intended, documented numerical change (then it invalidates published numbers) or a
refactor regression.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from merge_and_rebase.rebase.discrete_layer_match import DiscreteLayerPairing
from merge_and_rebase.rebase.methods._ariadne.alignment import apply_depth_pairing_override, compute_desired_effects
from merge_and_rebase.rebase.methods._ariadne.capture import capture_paired_boundary_activations
from merge_and_rebase.rebase.methods._ariadne.config import DirectResidualConfig
from merge_and_rebase.rebase.methods._ariadne.fit import fit_direct_residual
from merge_and_rebase.rebase.methods._ariadne.streaming import (
    fit_direct_residual_streaming,
    prepare_direct_residual_streaming,
)

from ._hashing import deterministic_cpu, flatten_tensors, hash_json, hash_tensor_dict

# --------------------------------------------------------------------------------------
# Expected hashes (captured at 7a7ee9a, CPU, torch.set_num_threads(1), deterministic algos).
# --------------------------------------------------------------------------------------
EXPECTED: dict[str, str] = {
    "bico_gradin_vision_fc": "4b121cac8c44864123a0f13437199be901082960669c91f705d965e25701d896",
    "bico_vision_fc": "c2ce523b8acc67770595cee94c0743591546615452bb058017495d2b886e173d",
    "brace_decoder_class_api:extend:base_state": "e1aed95cfd7b20c7491faf29512838fe3943e56ea1a75e42847f3715d67165a3",
    "brace_decoder_class_api:extend:ft_state": "f69a0d35d9478fc0aa9fffb62657623f151c877c90c04d1d78e9aafb825e432f",
    "brace_decoder_class_api:extend:task_vector": "9a38e272c443a77474fcbd4226c994696641bbe521ecd5937c0400ebaba6f9f2",
    "brace_decoder_class_api:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_decoder_class_api:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_decoder_class_api:shrink:task_vector": "19c5b2fd91a8d39abe398a4e7ab8b7a2177fbd80f6d66170cd03bd07c891e69b",
    "brace_decoder_independent:extend:base_state": "e1aed95cfd7b20c7491faf29512838fe3943e56ea1a75e42847f3715d67165a3",
    "brace_decoder_independent:extend:ft_state": "f69a0d35d9478fc0aa9fffb62657623f151c877c90c04d1d78e9aafb825e432f",
    "brace_decoder_independent:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_decoder_independent:extend:task_vector": "9a38e272c443a77474fcbd4226c994696641bbe521ecd5937c0400ebaba6f9f2",
    "brace_decoder_independent:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_decoder_independent:shrink:ft_state": "0518b97c9314e15c843bd9bb1547b55dbd6093567302cb13c97ed21aaeda82f6",
    "brace_decoder_independent:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_decoder_independent:shrink:task_vector": "19c5b2fd91a8d39abe398a4e7ab8b7a2177fbd80f6d66170cd03bd07c891e69b",
    "brace_decoder_shared:extend:base_state": "e1aed95cfd7b20c7491faf29512838fe3943e56ea1a75e42847f3715d67165a3",
    "brace_decoder_shared:extend:ft_state": "30b0958795e79dcf9a6b2e9b568f239a21e68b9633a40d953070161905270112",
    "brace_decoder_shared:extend:layout": "9e1555f3dbafa6ffd033b3124da7e76dd677700de308e1ab939be6795d4ac3f8",
    "brace_decoder_shared:extend:task_vector": "85ed569795f5bf2c9a12a6bf6638310379c46346d1307a87b934a1a8ea575849",
    "brace_decoder_shared:shrink:base_state": "056d5a27ddf6a98c4f8eeae6d22dba76ca0e8ad6791c6720567169733ff6741b",
    "brace_decoder_shared:shrink:ft_state": "82409d5fd7c4cbe5eacb89ab07dad9cb20c95ace83447cacbed012c59b59730b",
    "brace_decoder_shared:shrink:layout": "9756faed9b78a9078e554f9762b3759ad5cc1057d1a41b030acc72653e5fc028",
    "brace_decoder_shared:shrink:task_vector": "dcfdd8db68ee8378c0c4d522dbc5946cc61227a93115ca26e5f102e7f93d1b0d",
    "brace_vision_class_api:extend:base_state": "6c222cf231d2d5f75fd49e3dc91a3b97fc678e98ec1b10291a2f803cc8bd0fc6",
    "brace_vision_class_api:extend:ft_state": "3ad36b00d2f76bf4e2d90b451ab9ec536567be196be9915bde45e33483f202a4",
    "brace_vision_class_api:extend:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_vision_class_api:extend:task_vector": "3459b8aedcd04af883280b353b25dece88a9cc9e8e8c3c6a4b488c8f1e9000bd",
    "brace_vision_class_api:shrink:base_state": "0d521ec5d4c307fe2fbcd9648712f6eeb63dcac5906bdd2a9e0c3c5eab05ad4b",
    "brace_vision_class_api:shrink:ft_state": "2ee6262cc64ef4da09851ac7ebc7e90e2e9db8d3b28e3fd9dd00f31f546b063a",
    "brace_vision_class_api:shrink:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_vision_class_api:shrink:task_vector": "600fadb5341c4b9dd9ff50d0acfd73ecfd6753a39f1e11775ef98b7b3249d31a",
    "brace_vision_extend_duplicate_independent:base_state": "4a3edd070afa6742fea1da2c07cc5bc5d1cb2756990b3416973a0a027b1a4e88",
    "brace_vision_extend_duplicate_independent:ft_state": "54c7348ece68a3e95d663e29cde1097925cee15881ee0c4b70847357d5cfb1fa",
    "brace_vision_extend_duplicate_independent:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_vision_extend_duplicate_independent:task_vector": "e4c6e223f68f77bbbc42d841194b7cd676104a1ec77c3fc562c4a7cf369c3e93",
    "brace_vision_extend_duplicate_shared:base_state": "4a3edd070afa6742fea1da2c07cc5bc5d1cb2756990b3416973a0a027b1a4e88",
    "brace_vision_extend_duplicate_shared:ft_state": "cefaf964a7baffecbd42f63875095520fae593a075ca373a632211d1d731a124",
    "brace_vision_extend_duplicate_shared:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_vision_extend_duplicate_shared:task_vector": "34becee22c617335cd051b04662b9dd4f9de8cfcfcec2a96b2470ebc15d2353f",
    "brace_vision_extend_duplicate_shared_ft:base_state": "0e0e38d4775198988cd9ece1002cffafe77a2619e564cf3962dff8eb42568369",
    "brace_vision_extend_duplicate_shared_ft:ft_state": "54c7348ece68a3e95d663e29cde1097925cee15881ee0c4b70847357d5cfb1fa",
    "brace_vision_extend_duplicate_shared_ft:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_vision_extend_duplicate_shared_ft:task_vector": "ea4cf1c18a24400fae3a186195eefefbc4db649d4d08b6e65fbdb369ca951346",
    "brace_vision_extend_interpolate_independent:base_state": "d2927777ed73705995f8662f0285bb21da478e8d582455a784e1d33ee5a5a7f3",
    "brace_vision_extend_interpolate_independent:ft_state": "6ae5584333dc8265eec8ad14d47710c784f8e5edfdd7162c3dfa67a08482a42f",
    "brace_vision_extend_interpolate_independent:layout": "c2b0cbd3867b8f23accb5c96b5ae2f915cc3816a0aa0dc54cb56c0c4f070cf51",
    "brace_vision_extend_interpolate_independent:task_vector": "18b42ed4476040e02df930f6ce29d36fadf40a6317523fba2cdc8c7206b540f8",
    "brace_vision_shrink_interpolate_independent:base_state": "cf08855089d2dbe24b16c73343adc6f3bc983e9744d0b957e9ea647f84205334",
    "brace_vision_shrink_interpolate_independent:ft_state": "31abfad7eea8cc938eb7200fa24ad8fde587f0c2614d235adcd3264ebca255b0",
    "brace_vision_shrink_interpolate_independent:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_vision_shrink_interpolate_independent:task_vector": "1afc337731bd4e81e4ca6c11a4499fa916a4fdfebd26ba2f8e8d79cd8303fa93",
    "brace_vision_shrink_interpolate_shared:base_state": "cf08855089d2dbe24b16c73343adc6f3bc983e9744d0b957e9ea647f84205334",
    "brace_vision_shrink_interpolate_shared:ft_state": "19ec6af8c7cd4de95a2d2a9b683c43bbc09290e162780eaed618870ce4c051d5",
    "brace_vision_shrink_interpolate_shared:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_vision_shrink_interpolate_shared:task_vector": "3b8f2d5e6301f26e92116a4ec174a702122cc710d68042093f973d570af8894f",
    "brace_vision_shrink_interpolate_shared_ft:base_state": "56169469a92385efd720e719880d8da79b2220f92cd2b34981b5bc58db0f2e4d",
    "brace_vision_shrink_interpolate_shared_ft:ft_state": "31abfad7eea8cc938eb7200fa24ad8fde587f0c2614d235adcd3264ebca255b0",
    "brace_vision_shrink_interpolate_shared_ft:layout": "d5ce7bc70e71528c76d7b144a56b5f1c05bb0c59063b81f62f0cb59bac7c2a59",
    "brace_vision_shrink_interpolate_shared_ft:task_vector": "60ced8b4cb1274fb82a04fff3ca090e38a3ac23444badb371da365f60590968c",
    "dr_alignment_ridge_streaming:extend:alignment": "2c1b900133f2832ad56a510ea0a8bc51c8891f3c1e68131199eab426dcd594d1",
    "dr_alignment_ridge_streaming:extend:task_vector": "0b9d5a3ca8bf2470d00daf3a27954a738e70c7adc2587cde0cdd566e3ffe1762",
    "dr_alignment_ridge_streaming:shrink:alignment": "d68ec69ff895eda8cce861710365359b8d58fecddda0d52931d8b6e669154b23",
    "dr_alignment_ridge_streaming:shrink:task_vector": "91e4e20a3e5a92f19b280bf8a4611bc407073ec4e91b81be3616ea6df204011d",
    "dr_block_split_backfit:extend:alignment": "e6ee902c3d40e0511bf7ad27c63e71532b2f922f52e50faea5cb1c149fe7ef6c",
    "dr_block_split_backfit:extend:task_vector": "0db607aa29e80100c7096457b3596f53f4d67a54c7335d2d9610a287213a03db",
    "dr_block_split_backfit:shrink:alignment": "1f687e3e452feaefbab1a7bc89de3886c8d14f2d6a6c5cf5d4cba0f5a2a121e7",
    "dr_block_split_backfit:shrink:task_vector": "39818ce857714640737e9805be874ce2cf301a1afce2b93be66288254e9581f4",
    "dr_block_split_joint:extend:alignment": "e6ee902c3d40e0511bf7ad27c63e71532b2f922f52e50faea5cb1c149fe7ef6c",
    "dr_block_split_joint:extend:task_vector": "fc02aac2fcbbc611c284d7e9840ee06490769e2e7648566836af4f6a44949688",
    "dr_block_split_joint:shrink:alignment": "1f687e3e452feaefbab1a7bc89de3886c8d14f2d6a6c5cf5d4cba0f5a2a121e7",
    "dr_block_split_joint:shrink:task_vector": "91416a6fe342a1c0724b12a39fa76b4ca1a44137459fa86f35cbc0046b6fc721",
    "dr_default_resident_fixed_relative:extend:alignment": "e6ee902c3d40e0511bf7ad27c63e71532b2f922f52e50faea5cb1c149fe7ef6c",
    "dr_default_resident_fixed_relative:extend:task_vector": "113c485c5be7790e5b7cd0e57a11ed66392eabc5c9376dd8a58bb7053f77b399",
    "dr_default_resident_fixed_relative:shrink:alignment": "1f687e3e452feaefbab1a7bc89de3886c8d14f2d6a6c5cf5d4cba0f5a2a121e7",
    "dr_default_resident_fixed_relative:shrink:task_vector": "208eaae4c7fac3efbcad60c734e05894e575b08d5751e2be52ff02a1935d8ad9",
    "dr_default_streaming_fixed_relative:extend:alignment": "8255b82b82472d08ce56fb36caf5d087205d76ccfcf007217c09926dd7eab433",
    "dr_default_streaming_fixed_relative:extend:task_vector": "113c485c5be7790e5b7cd0e57a11ed66392eabc5c9376dd8a58bb7053f77b399",
    "dr_default_streaming_fixed_relative:shrink:alignment": "552f60a1ce474bb04de0283f4f97310c03886d8b19692af0996ea31303b77966",
    "dr_default_streaming_fixed_relative:shrink:task_vector": "208eaae4c7fac3efbcad60c734e05894e575b08d5751e2be52ff02a1935d8ad9",
    "dr_depth_pairing_reversed:extend:alignment": "e4be8b037eddd579270b94435b0e4e902a79366b12f9221177f98a2dd3b05013",
    "dr_depth_pairing_reversed:extend:task_vector": "51dc6aeffd968768a576e5d66454b06bf40f9f972f7c7f16e31a31178704f43f",
    "dr_depth_pairing_reversed:shrink:alignment": "a1d7f0b197b7a3d7c21c2a5e20a04b3225fa3151f7419890c2db229ba157af7c",
    "dr_depth_pairing_reversed:shrink:task_vector": "7c9eabb3a13918253fb2d223e658e17e8280f9364cd4360c06e58f9cfc080b74",
    "dr_depth_pairing_reversed_streaming:extend:alignment": "c4b0787bbc478e5e0e24ca34e05318538289b9990993a13b1999424177c259b9",
    "dr_depth_pairing_reversed_streaming:extend:task_vector": "51dc6aeffd968768a576e5d66454b06bf40f9f972f7c7f16e31a31178704f43f",
    "dr_depth_pairing_reversed_streaming:shrink:alignment": "9358fcceb6ee6eee77e45365cacfb103f7271f2ede3ede96cc780a4c6b11e83f",
    "dr_depth_pairing_reversed_streaming:shrink:task_vector": "7c9eabb3a13918253fb2d223e658e17e8280f9364cd4360c06e58f9cfc080b74",
    "dr_gradient_procrustes_resident:extend:alignment": "9d7a320d88482f84572160051b6ac6b2e48823eadf2493c5ee0dd0d4e628172b",
    "dr_gradient_procrustes_resident:extend:task_vector": "bcab85765349a48c887cd9785ec4c5411f8115165d57058f31a5d47e9023c7cc",
    "dr_gradient_procrustes_resident:shrink:alignment": "ce8025cec6f316fac2888f38f1d53211056a55ec1b076f28448e57f9a5827412",
    "dr_gradient_procrustes_resident:shrink:task_vector": "5d8821a33528782fdc6458f347583c8f60d093325dcae4cace0df246252419ad",
    "dr_gradient_procrustes_streaming:extend:alignment": "2730f8a9ef810d35bb5678a3d64a3829613d7a11fd9bb8df57ec1ca0272880ff",
    "dr_gradient_procrustes_streaming:extend:task_vector": "214018ae5b0e75e051c8c2956dde958d35db418bc80c53849a0911c7740a6c58",
    "dr_gradient_procrustes_streaming:shrink:alignment": "2771b082b1e2de769090edbc72127f895a406f63dee9bc7f18f63f976b88e857",
    "dr_gradient_procrustes_streaming:shrink:task_vector": "db74ab5b9307b9c548822aad5f67bd6b6e320f5c8da76b13c01d3021230cf979",
    "dr_main_cproj_resident_eb:extend:alignment": "e6ee902c3d40e0511bf7ad27c63e71532b2f922f52e50faea5cb1c149fe7ef6c",
    "dr_main_cproj_resident_eb:extend:task_vector": "98ef3bc1592b30e952712731bece8536da3f9cdd7841f908109c70e8603781cb",
    "dr_main_cproj_resident_eb:shrink:alignment": "1f687e3e452feaefbab1a7bc89de3886c8d14f2d6a6c5cf5d4cba0f5a2a121e7",
    "dr_main_cproj_resident_eb:shrink:task_vector": "39e63783763f302dc635be0eec9f744bd3e8798970168aa5b355e9338ca02f6f",
    "dr_main_cproj_streaming_eb:extend:alignment": "8255b82b82472d08ce56fb36caf5d087205d76ccfcf007217c09926dd7eab433",
    "dr_main_cproj_streaming_eb:extend:task_vector": "98ef3bc1592b30e952712731bece8536da3f9cdd7841f908109c70e8603781cb",
    "dr_main_cproj_streaming_eb:shrink:alignment": "552f60a1ce474bb04de0283f4f97310c03886d8b19692af0996ea31303b77966",
    "dr_main_cproj_streaming_eb:shrink:task_vector": "39e63783763f302dc635be0eec9f744bd3e8798970168aa5b355e9338ca02f6f",
    "dr_main_cproj_streaming_fixed_relative:extend:alignment": "8255b82b82472d08ce56fb36caf5d087205d76ccfcf007217c09926dd7eab433",
    "dr_main_cproj_streaming_fixed_relative:extend:task_vector": "06f9ab188bd7a680abb045a43b0dba6d069fa1bf31cd3c5f9fbd20e449006202",
    "dr_main_cproj_streaming_fixed_relative:shrink:alignment": "552f60a1ce474bb04de0283f4f97310c03886d8b19692af0996ea31303b77966",
    "dr_main_cproj_streaming_fixed_relative:shrink:task_vector": "ddb76c6e2e142f0e0d514c5f5fe58111ed131d808fdf3d6347301e28136e1e4c",
    "dr_od_streaming_eb:extend:alignment": "8255b82b82472d08ce56fb36caf5d087205d76ccfcf007217c09926dd7eab433",
    "dr_od_streaming_eb:extend:task_vector": "c472d2b55cd53b7230bc37ab872e7cdddb6e49f1cc5f1dd9c02229b806e58b0b",
    "dr_od_streaming_eb:shrink:alignment": "552f60a1ce474bb04de0283f4f97310c03886d8b19692af0996ea31303b77966",
    "dr_od_streaming_eb:shrink:task_vector": "9f759855432ed43cd619b11eb35d487f696f2dcba5848520d922f2918376c342",
    "dr_orchestration_main_streaming_eb:extend:summary": "1686a7afde96ab90d0ca062e69f28d0a75a3f51f09f79db7b275e2573aa6c7ec",
    "dr_orchestration_main_streaming_eb:extend:task_vector": "98ef3bc1592b30e952712731bece8536da3f9cdd7841f908109c70e8603781cb",
    "dr_orchestration_main_streaming_eb:shrink:summary": "0a4c6c11ff8ac3619b7f11bb0d835de0fd8006c8329e9727f4d1ad28673b8984",
    "dr_orchestration_main_streaming_eb:shrink:task_vector": "39e63783763f302dc635be0eec9f744bd3e8798970168aa5b355e9338ca02f6f",
    "dr_orchestration_od_resident_fixed_relative:extend:summary": "ba8c4447a4f3aba99a2640c8728eb5aaa8c055441ba8dafffcb6ef7efbcb7d6a",
    "dr_orchestration_od_resident_fixed_relative:extend:task_vector": "113c485c5be7790e5b7cd0e57a11ed66392eabc5c9376dd8a58bb7053f77b399",
    "dr_orchestration_od_resident_fixed_relative:shrink:summary": "82656c63449b965c653663f63107e1a47ea0dfc86c8e76179b3dfecafa9be8ee",
    "dr_orchestration_od_resident_fixed_relative:shrink:task_vector": "208eaae4c7fac3efbcad60c734e05894e575b08d5751e2be52ff02a1935d8ad9",
    "dr_random_isometry_resident:extend:alignment": "9d77ebb184f6ac41583b52fea44d1ab9b12f9ba5c9fb4668a57d619b2af3e8da",
    "dr_random_isometry_resident:extend:task_vector": "67cf0ee866c51175f41ba8838b85b19be6b6f097857b33de4182c9cf7df07067",
    "dr_random_isometry_resident:shrink:alignment": "5f4a4c43c168f6fd2307862425a94a60e7a0bd61d562b0c09c8ff1e67dc7edab",
    "dr_random_isometry_resident:shrink:task_vector": "d5839f8624b04da2713fa2b2cb77001551610137fac7759f23082d2c94486922",
    "dr_random_isometry_streaming:extend:alignment": "2e894bd50a1f4a1d3f04d2e6c81dd1fd347901a454fc0d0588bdb21b7cfa40b8",
    "dr_random_isometry_streaming:extend:task_vector": "67cf0ee866c51175f41ba8838b85b19be6b6f097857b33de4182c9cf7df07067",
    "dr_random_isometry_streaming:shrink:alignment": "f537d10f91a1936d735ced2dfc92f0ec7b423a7a6e702285a2d10199bc9ba4f2",
    "dr_random_isometry_streaming:shrink:task_vector": "d5839f8624b04da2713fa2b2cb77001551610137fac7759f23082d2c94486922",
    "llm_rebase_bico:extend:brace_and_delta": "2db18d9874516aabed405bae08e9a26e2ed8ba2e834a5755d1eee0c6ba5f0f74",
    "llm_rebase_bico:extend:transported_delta": "951710ff5e19b3d292f6a0b1da931231727c8d5a32bed6be3cfc7c50afd94ac4",
    "llm_rebase_bico:shrink:brace_and_delta": "ad871adbc4c5aa6ac60b7c676e9dc84561a295c7ed007966fdd4b2a65f5ad548",
    "llm_rebase_bico:shrink:transported_delta": "90616aa0dbdcb9a6e222aa2febe7f0da671327634066b2992c4b257bfa52d714",
    "llm_rebase_theseus:extend:brace_and_delta": "2db18d9874516aabed405bae08e9a26e2ed8ba2e834a5755d1eee0c6ba5f0f74",
    "llm_rebase_theseus:extend:transported_delta": "e8df06df8fa23554912515fb4cf84ac00f661c92a127a183a12bebe442047e50",
    "llm_rebase_theseus:shrink:brace_and_delta": "ad871adbc4c5aa6ac60b7c676e9dc84561a295c7ed007966fdd4b2a65f5ad548",
    "llm_rebase_theseus:shrink:transported_delta": "a86865c93e12a8c0d76dccc4cbc93ba432ed09dadfe11236230f9c07ba0a5ed1",
    "llm_rebase_theseus_gqa:extend:brace_and_delta": "2db18d9874516aabed405bae08e9a26e2ed8ba2e834a5755d1eee0c6ba5f0f74",
    "llm_rebase_theseus_gqa:extend:transported_delta": "cb7dc0c02af47403911cc89157fb31cb8fe136d36b5b2e70c3bd9ad799190288",
    "llm_rebase_theseus_gqa:shrink:brace_and_delta": "ad871adbc4c5aa6ac60b7c676e9dc84561a295c7ed007966fdd4b2a65f5ad548",
    "llm_rebase_theseus_gqa:shrink:transported_delta": "bc26de69ec73f1589e7ca01af20ca22b640240bc0a4a556b7208b377f4cd5579",
    "theseus_vision_fc": "5ad4876a13fa9c48d39a1d97d781b47145d462b5e03ac863072e5a069d90b8f1",
    "theseus_vision_fc_centered": "a2a94edf84da6a463282fc85c6ce25c64d375c08b7bf8375aa22157a8ffbc958",
    "theseus_vision_fc_data_free": "23e9a0a6b41ad9c5ee11f5779e77e7104971bba27be4159c84651afe16b0e28b",
    "theseus_vision_fc_whiten025": "aee27268f56e8fce14d2c1b22e891f2b7fd91302a6d2d162a6f9687fce8ea1a5",
    "vision_rebase_bico_brace_prestep:extend:brace_and_delta": "9d54bc803bd6881986f46e8d87d9e81f762ac631c09ee9b1d922c2eb8a6b8b71",
    "vision_rebase_bico_brace_prestep:extend:transported_delta": "a6c3e079cc287a58aa4778b792cce308d54cf1b23c991e66c133968c3b9dba38",
    "vision_rebase_bico_brace_prestep:shrink:brace_and_delta": "12abe7a8ede5340e42316adea34ed00da12e2c3fe1ebaab75fd6e5d2af676b0a",
    "vision_rebase_bico_brace_prestep:shrink:transported_delta": "6bcebcedeb4398b4251547927e29dc0c9abb8684cfe917bf93b17e1f2f5f7b8a",
    "vision_rebase_bico_no_prestep:samedepth:brace_and_delta": "1fe61eed751c59c39920bfe4e530af9bbae8cc00ab3132bd1c4e54f0dcb14647",
    "vision_rebase_bico_no_prestep:samedepth:transported_delta": "f1aeaf7045945a63e8cda60ab9a1d6cc730932e95d3a82e0229fd71446a76e01",
    "vision_rebase_theseus_brace_prestep:extend:brace_and_delta": "9d54bc803bd6881986f46e8d87d9e81f762ac631c09ee9b1d922c2eb8a6b8b71",
    "vision_rebase_theseus_brace_prestep:extend:transported_delta": "205a2d3a5cd14f036c3f125a8db16161cd69f546e24f5c828027d45519fb57ba",
    "vision_rebase_theseus_brace_prestep:shrink:brace_and_delta": "12abe7a8ede5340e42316adea34ed00da12e2c3fe1ebaab75fd6e5d2af676b0a",
    "vision_rebase_theseus_brace_prestep:shrink:transported_delta": "d5db7ff6e25218727f88a85543dca479c172ea4afc7589fc25249c675f964b7d",
    "vision_rebase_theseus_no_prestep:samedepth:brace_and_delta": "1fe61eed751c59c39920bfe4e530af9bbae8cc00ab3132bd1c4e54f0dcb14647",
    "vision_rebase_theseus_no_prestep:samedepth:transported_delta": "9a71d4f4531618193315ae6547580fa29076f057285c3beaed20da39ef6771e4",
}


@pytest.fixture(autouse=True)
def _deterministic():
    with deterministic_cpu(seed=0):
        yield


def _check(name: str, actual: str) -> None:
    capture = os.environ.get("GOLDEN_CAPTURE")
    if capture:  # (re)generation aid only: GOLDEN_CAPTURE=<file> appends "name hash" lines and skips the assert
        with open(capture, "a") as fh:
            fh.write(f"{name} {actual}\n")
        return
    assert name in EXPECTED, f"no expected hash recorded for {name!r} (actual {actual})"
    assert actual == EXPECTED[name], f"{name}: golden hash changed\n  expected {EXPECTED[name]}\n  actual   {actual}"


# --------------------------------------------------------------------------------------
# Direct Residual (a.k.a. Ariadne) -- hand-built block fixture, copied from
# tests/test_direct_residual_ablation_v2_golden_hashes_20260925.py (same seeds, so the
# "default" case below must reproduce that file's pinned hashes).
# --------------------------------------------------------------------------------------


class _Attention(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.out_proj(x)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = _Attention(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 2)),
                    ("gelu", torch.nn.GELU()),
                    ("c_proj", torch.nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ls_2 = torch.nn.Identity()

    def forward(self, x):
        return x + self.attn(x) + self.ls_2(self.mlp(x))


class _Visual(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = torch.nn.Linear(4, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width) for _ in range(depth)])

    def forward(self, images):
        x = self.input(images)
        for block in self.transformer.resblocks:
            x = block(x)
        return x.mean(dim=1)


class _Model(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.visual = _Visual(width, depth)

    def encode_image(self, x):
        return self.visual(x)


def _dr_loader(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 5, 4, generator=generator)
    return DataLoader(TensorDataset(images, torch.arange(n) % 6), batch_size=2, shuffle=False)


def _dr_tuned_copy(model, seed, scale=0.2):
    tuned = deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in tuned.visual.transformer.resblocks:
            block.mlp.c_proj.weight.add_(scale * torch.randn_like(block.mlp.c_proj.weight))
            block.attn.out_proj.weight.add_(scale * torch.randn_like(block.attn.out_proj.weight))
    return tuned


def _dr_setup(source_depth, target_depth, width=5, seed=11):
    torch.manual_seed(seed)
    source_base = _Model(width, source_depth).eval()
    target_base = _Model(width, target_depth).eval()
    source_ft = _dr_tuned_copy(source_base, seed + 1)
    data = _dr_loader(seed=seed + 2)
    pairing = DiscreteLayerPairing.compute(source_depth, target_depth)
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    return source_base, source_ft, target_base, (data, data), pairing, target_base_sd


DEPTHS = {"extend": (2, 4), "shrink": (4, 2)}


def _dr_kernel(overrides: dict, direction: str, *, depth_pairing: str = "relative", setup=None, recipes=(None, None)):
    """Run capture -> alignment -> fit on the public DR kernel API.

    Returns ``(task_vector, alignment)``: the fitted task-vector dict and the alignment
    maps / desired effects the fit was built on, both ``dict[str, Tensor]``.
    """
    if setup is None:
        setup = _dr_setup(*DEPTHS[direction])
    source_base, source_ft, target_base, (source_loader, target_loader), pairing, target_base_sd = setup
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, **overrides)
    pairing = apply_depth_pairing_override(pairing, depth_pairing)
    source_recipe, target_recipe = recipes
    align_kwargs = dict(
        procrustes_source=config.procrustes_source,
        alignment_map=config.alignment_map,
        alignment_row_weighting=config.alignment_row_weighting,
        alignment_seed=config.alignment_seed,
    )
    if config.activation_storage == "streaming":
        prepared = prepare_direct_residual_streaming(
            source_base,
            target_base,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device="cpu",
            source_ft_model=source_ft,
            source_recipe=source_recipe,
            target_recipe=target_recipe,
            **align_kwargs,
        )
        corr, _rows = fit_direct_residual_streaming(
            target_base, target_base_sd, source_base, source_ft, prepared, pairing, config=config, device="cpu"
        )
        alignment = {f"q/{j}": q for j, q in prepared["q_by_position"].items()}
    else:
        captured = capture_paired_boundary_activations(
            source_base,
            source_ft,
            target_base,
            source_loader,
            target_loader,
            pairing,
            num_batches=config.num_batches,
            seed=config.seed,
            device="cpu",
            procrustes_source=config.procrustes_source,
            source_recipe=source_recipe,
            target_recipe=target_recipe,
        )
        align_diag: dict = {}
        desired = compute_desired_effects(
            captured, pairing, residual_target=config.residual_target, diagnostics_out=align_diag, **align_kwargs
        )
        corr, _rows = fit_direct_residual(
            target_base, target_base_sd, captured, desired, pairing, config=config, device="cpu"
        )
        alignment = flatten_tensors({"desired": desired, "q": {j: d["q"] for j, d in align_diag.items()}})
    assert corr, "empty task vector would pin nothing"
    return corr, alignment


_MAIN = dict(
    components=("mlp.c_proj",),
    component_target="block_boundary",
    activation_storage="streaming",
    ridge_estimator="empirical_bayes",
    alignment_map="polar",
    procrustes_source="activation",
)
_OD = {**_MAIN, "components": ("attn.out_proj", "mlp.c_proj")}
_RESIDENT_MAIN = {**_MAIN, "activation_storage": "resident"}
_DEFAULT = {}  # DirectResidualConfig defaults: O+D, fixed_relative, resident, polar, activation
_JOINT = dict(block_split="joint")
_BACKFIT = dict(block_split="backfit", backfit_max_iters=2)
_RANDOM_ISOMETRY = {**_MAIN, "alignment_map": "random_isometry", "alignment_seed": 3}

DR_CASES = {
    # name: (config overrides, depth_pairing)
    "dr_main_cproj_streaming_eb": (_MAIN, "relative"),
    "dr_main_cproj_streaming_fixed_relative": ({**_MAIN, "ridge_estimator": "fixed_relative"}, "relative"),
    "dr_alignment_ridge_streaming": ({**_MAIN, "alignment_map": "ridge"}, "relative"),
    "dr_od_streaming_eb": (_OD, "relative"),
    "dr_main_cproj_resident_eb": (_RESIDENT_MAIN, "relative"),
    "dr_default_resident_fixed_relative": (_DEFAULT, "relative"),
    "dr_default_streaming_fixed_relative": (dict(activation_storage="streaming"), "relative"),
    "dr_block_split_joint": (_JOINT, "relative"),
    "dr_block_split_backfit": (_BACKFIT, "relative"),
    "dr_random_isometry_streaming": (_RANDOM_ISOMETRY, "relative"),
    "dr_random_isometry_resident": ({**_RANDOM_ISOMETRY, "activation_storage": "resident"}, "relative"),
    "dr_depth_pairing_reversed": ({**_MAIN, "activation_storage": "resident"}, "reversed"),
    "dr_depth_pairing_reversed_streaming": (_MAIN, "reversed"),
}


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize("case", sorted(DR_CASES))
def test_direct_residual_kernel(case, direction):
    overrides, depth_pairing = DR_CASES[case]
    tv, alignment = _dr_kernel(overrides, direction, depth_pairing=depth_pairing)
    _check(f"{case}:{direction}:task_vector", hash_tensor_dict(tv))
    _check(f"{case}:{direction}:alignment", hash_tensor_dict(alignment))


# --------------------------------------------------------------------------------------
# procrustes_source="gradient": needs a real open_clip VisionTransformer (the gradient
# bank hooks the real block outputs). Fixture adapted from
# tests/test_direct_residual_gradient_procrustes.py; widths/layers are minimal.
# --------------------------------------------------------------------------------------


class _IdentityTensorDataset(TensorDataset):
    def __init__(self, images, labels, sample_ids):
        super().__init__(images, labels)
        self.sample_ids = sample_ids


class _DummyClassifier:
    normalize = True

    def _compute_zeroshot_text_features(self, *_args, **_kwargs):
        raise AssertionError("text_features must be supplied explicitly")


_GRAD_SPEC = {
    "extend": dict(
        source=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
        target=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3),
    ),
    "shrink": dict(
        source=dict(image_size=24, patch_size=4, width=12, layers=4, heads=3),
        target=dict(image_size=16, patch_size=4, width=8, layers=2, heads=2),
    ),
}


def _grad_setup(direction, seed=101, n=6):
    from open_clip.transformer import VisionTransformer

    from merge_and_rebase.models.grad_recipes import clip_contrastive_recipe

    class _CLIPLike(torch.nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual
            self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

        def encode_image(self, x):
            return self.visual(x)

    def make_vit(*, image_size, patch_size, width, layers, heads, seed):
        torch.manual_seed(seed)
        vt = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=2.0,
            ls_init_value=None,
            output_dim=width,
            pool_type="tok",
        )
        return _CLIPLike(vt).eval()

    def loader(image_size, loader_seed, ids):
        generator = torch.Generator().manual_seed(loader_seed)
        images = torch.randn(n, 3, image_size, image_size, generator=generator)
        return DataLoader(_IdentityTensorDataset(images, torch.arange(n), ids), batch_size=2, shuffle=False)

    spec = _GRAD_SPEC[direction]
    source_base = make_vit(seed=seed, **spec["source"])
    target_base = make_vit(seed=seed + 1, **spec["target"])
    source_ft = _dr_tuned_copy(source_base, seed + 2)
    ids = [str(i) for i in range(n)]
    source_loader = loader(spec["source"]["image_size"], seed + 3, ids)
    target_loader = loader(spec["target"]["image_size"], seed + 4, ids)
    pairing = DiscreteLayerPairing.compute(spec["source"]["layers"], spec["target"]["layers"])
    target_base_sd = {k: v.clone() for k, v in target_base.state_dict().items()}
    source_text = torch.randn(n, spec["source"]["width"])
    target_text = torch.randn(n, spec["target"]["width"])
    recipes = (
        clip_contrastive_recipe(_DummyClassifier(), [], None, text_features=source_text, device="cpu"),
        clip_contrastive_recipe(_DummyClassifier(), [], None, text_features=target_text, device="cpu"),
    )
    setup = (source_base, source_ft, target_base, (source_loader, target_loader), pairing, target_base_sd)
    return setup, recipes


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize("storage", ["resident", "streaming"])
def test_direct_residual_gradient_procrustes(storage, direction):
    setup, recipes = _grad_setup(direction)
    overrides = {**_MAIN, "activation_storage": storage, "procrustes_source": "gradient"}
    tv, alignment = _dr_kernel(overrides, direction, setup=setup, recipes=recipes)
    name = f"dr_gradient_procrustes_{storage}:{direction}"
    _check(f"{name}:task_vector", hash_tensor_dict(tv))
    _check(f"{name}:alignment", hash_tensor_dict(alignment))


# --------------------------------------------------------------------------------------
# Orchestration level, Direct Residual: vision_rebase._run_direct_residual_fit is the
# deepest entry that runs offline (main() needs real OpenCLIP checkpoints). It chains
# pairing -> capture -> alignment -> fit -> strength scaling and returns the task vector
# plus diagnostics, which are pinned with every timing / memory / path field removed.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize(
    ("case", "overrides"),
    [
        ("dr_orchestration_main_streaming_eb", _MAIN),
        ("dr_orchestration_od_resident_fixed_relative", {}),
    ],
)
def test_vision_rebase_run_direct_residual_fit(case, overrides, direction):
    from merge_and_rebase.eval.vision_rebase import _run_direct_residual_fit

    source_base, source_ft, target_base, (source_loader, target_loader), pairing, target_base_sd = _dr_setup(
        *DEPTHS[direction]
    )
    config = DirectResidualConfig(num_batches=3, ridge_relative=0.05, **overrides)
    delta, timing, diagnostics, extra = _run_direct_residual_fit(
        source_base_model=source_base,
        source_ft_model=source_ft,
        target_model=target_base,
        target_base_sd=target_base_sd,
        source_loader=source_loader,
        target_loader=target_loader,
        pairing=pairing,
        config=config,
        device="cpu",
    )
    assert delta and set(timing) == {"alignment_calibration", "correction_fit", "cost_phases"}
    # `timing` is wall clock / memory only: deliberately not pinned.
    _check(f"{case}:{direction}:task_vector", hash_tensor_dict(delta))
    _check(f"{case}:{direction}:summary", hash_json({"diagnostics": diagnostics, "extra": extra}))


# --------------------------------------------------------------------------------------
# THESEUS / BiCo transport on tiny vision models (direct method API) and the
# orchestration path vision_rebase uses: BRACE prestep (run_block_extension) ->
# _build_rebase_prepared -> method.transport.
# --------------------------------------------------------------------------------------


class _FcVisual(torch.nn.Module):
    def __init__(self, in_dim=6, hid_dim=8, out_dim=5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hid_dim)
        self.ln = torch.nn.LayerNorm(hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.ln(self.fc1(x)))


class _FcModel(torch.nn.Module):
    def __init__(self, in_dim=6, hid_dim=8, out_dim=5):
        super().__init__()
        self.visual = _FcVisual(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)
        self.logit_scale = torch.nn.Parameter(torch.ones(1))

    def encode_image(self, x):
        return self.visual(x)


def _class_loader(n=16, in_dim=6, batch_size=4, seed=0, n_classes=5):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=generator)
    # Labels are independent of `seed`: THESEUS/BiCo require label-aligned source/target loaders.
    y = torch.randint(0, n_classes, (n,), generator=torch.Generator().manual_seed(1000))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _simple_recipe(model, batch):
    images, labels = batch
    loss = torch.nn.functional.cross_entropy(model.encode_image(images), labels)
    return loss, [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _visual_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _fc_transport(method_name, **kwargs):
    import merge_and_rebase.rebase.methods  # noqa: F401  (registers the methods)
    from merge_and_rebase.rebase.registry import get_method

    torch.manual_seed(5)
    source_model = _FcModel(hid_dim=8)
    target_model = _FcModel(hid_dim=7)
    source_base = _visual_state(source_model)
    target_base = _visual_state(target_model)
    gen = torch.Generator().manual_seed(6)
    delta = {
        k: torch.randn(v.shape, generator=gen)
        for k, v in source_base.items()
        if k.startswith("visual.") and v.is_floating_point()
    }
    loader = _class_loader(seed=7)
    extra = {}
    if method_name.startswith("bico"):
        extra = dict(source_recipe=_simple_recipe, target_recipe=_simple_recipe)
    transported = get_method(method_name).transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        source_dataloader=loader,
        target_dataloader=loader,
        device="cpu",
        seq_align="mean",
        num_batches=2,
        seed=123,
        strict=True,
        verbose=False,
        show_progress=False,
        **extra,
        **kwargs,
    )
    assert transported and set(transported) == set(delta)
    return transported


@pytest.mark.parametrize(
    ("case", "method_name", "kwargs"),
    [
        ("theseus_vision_fc", "theseus", {}),
        ("theseus_vision_fc_whiten025", "theseus", {"whiten_power": 0.25}),
        ("theseus_vision_fc_centered", "theseus", {"center_acts": True}),
        ("bico_vision_fc", "bico", {}),
        ("bico_gradin_vision_fc", "bico_gradin", {}),
    ],
)
def test_vision_transport_direct_method_api(case, method_name, kwargs):
    _check(case, hash_tensor_dict(_fc_transport(method_name, **kwargs)))


def test_theseus_vision_data_free_transport():
    import merge_and_rebase.rebase.methods  # noqa: F401
    from merge_and_rebase.rebase.registry import get_method

    torch.manual_seed(5)
    source_model, target_model = _FcModel(hid_dim=8), _FcModel(hid_dim=7)
    source_base, target_base = _visual_state(source_model), _visual_state(target_model)
    gen = torch.Generator().manual_seed(6)
    delta = {k: torch.randn(v.shape, generator=gen) for k, v in source_base.items() if k.startswith("visual.")}
    transported = get_method("theseus").transport(
        source_base=source_base,
        target_base=target_base,
        delta=delta,
        source_model=source_model,
        target_model=target_model,
        device="cpu",
        covariance_mode="data_free",
        whiten_power=0.25,
        strict=True,
        verbose=False,
        show_progress=False,
    )
    _check("theseus_vision_fc_data_free", hash_tensor_dict(transported))


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
    def __init__(self, depth, in_dim=6, width=4, out_dim=5):
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


def _build_vision_orchestration(method_name, source_depth, target_depth, *, prestep):
    """vision_rebase's per-task flow, minus checkpoint IO: BRACE prestep -> prepare -> transport."""
    from types import SimpleNamespace

    import merge_and_rebase.rebase.methods  # noqa: F401
    from merge_and_rebase.eval.vision_rebase import _build_rebase_prepared
    from merge_and_rebase.rebase.block_extension.config import BlockExtensionConfig
    from merge_and_rebase.rebase.block_extension.vision import run_block_extension
    from merge_and_rebase.rebase.registry import get_method

    torch.manual_seed(7)
    source_base = _AttnModel(source_depth, width=4)
    source_ft = deepcopy(source_base)
    with torch.no_grad():
        for p in source_ft.parameters():
            p.add_(0.05 * torch.randn_like(p))
    target = _AttnModel(target_depth, width=6)
    clf_source = SimpleNamespace(model=source_base, normalize=True)
    clf_target = SimpleNamespace(model=target, normalize=True)
    source_loader = _class_loader(n=8, batch_size=4, seed=8)
    target_loader = _class_loader(n=8, batch_size=4, seed=9)
    summary: dict = {}
    if prestep:
        extended_base, extended_ft = deepcopy(source_base), deepcopy(source_ft)
        run_block_extension(
            source_base_model=extended_base,
            source_ft_model=extended_ft,
            calibration_loader=source_loader,
            target_layers_total=target_depth,
            config=BlockExtensionConfig(
                extension_strategy="interpolate_per_weight",
                skip_correction=False,
                n_batches_act=2,
                verbose=False,
                show_progress=False,
            ),
            device="cpu",
        )
        source_base_model_task = extended_base
        base_sd, ft_sd = _visual_state(extended_base), _visual_state(extended_ft)
    else:
        source_base_model_task = None
        base_sd, ft_sd = _visual_state(source_base), _visual_state(source_ft)
    task_delta = {k: ft_sd[k] - v for k, v in base_sd.items() if k.startswith("visual.") and v.is_floating_point()}
    summary["brace_source_base"] = hash_tensor_dict(base_sd)
    summary["brace_source_ft"] = hash_tensor_dict(ft_sd)
    summary["task_delta"] = hash_tensor_dict(task_delta)
    method = get_method(method_name)
    method_params = {"seq_align": "mean", "num_batches": 2, "verbose": False, "show_progress": False}
    target_base_sd = _visual_state(target)
    text_features = torch.randn(5, 5, generator=torch.Generator().manual_seed(10))
    prepared = _build_rebase_prepared(
        method_name=method_name,
        method=method,
        method_params=dict(method_params),
        cfg={"seed": 33},
        device="cpu",
        grad_batch_size=None,
        grad_imgs_per_class=None,
        grad_num_batches=None,
        theseus_like_method=method_name.startswith("theseus"),
        bico_mode=method_name.startswith("bico"),
        run_block_extension_prestep=prestep,
        clf_source=clf_source,
        clf_target=clf_target,
        classnames=[],
        loaders=SimpleNamespace(train=target_loader),
        source_loaders=SimpleNamespace(train=source_loader),
        build_cfg_task=None,
        source_build_cfg_task=None,
        task_source_base_sd=base_sd,
        target_base_sd=target_base_sd,
        task_delta=task_delta,
        source_base_model_task=source_base_model_task,
        transfusion_prepared=None,
        source_text_features=text_features,
        target_text_features=text_features,
    )
    transported = method.transport(
        source_base=base_sd,
        target_base=target_base_sd,
        delta=task_delta,
        strict=True,
        prepared=prepared,
        **method_params,
    )
    assert transported
    return transported, summary


@pytest.mark.parametrize("method_name", ["theseus", "bico"])
@pytest.mark.parametrize(
    ("direction", "source_depth", "target_depth", "prestep"),
    [("extend", 1, 2, True), ("shrink", 2, 1, True), ("samedepth", 2, 2, False)],
)
def test_vision_rebase_orchestration_with_brace_prestep(method_name, direction, source_depth, target_depth, prestep):
    transported, summary = _build_vision_orchestration(method_name, source_depth, target_depth, prestep=prestep)
    case = f"vision_rebase_{method_name}_{'brace_prestep' if prestep else 'no_prestep'}:{direction}"
    _check(f"{case}:brace_and_delta", hash_json(summary))
    _check(f"{case}:transported_delta", hash_tensor_dict(transported))


# --------------------------------------------------------------------------------------
# BRACE vision: BlockExtender via run_block_extension (and the class directly).
# Fixture adapted from tests/test_block_extension_prestep.py. Hashed: both endpoints'
# state dicts after the resize, their task vector (ft - base) and the realized layout.
# --------------------------------------------------------------------------------------


class _BrAttn(torch.nn.Module):
    """Fused-in_proj attention exposing the attributes BRACE's capture hooks read."""

    def __init__(self, dim):
        super().__init__()
        self.in_proj_weight = torch.nn.Parameter(torch.randn(3 * dim, dim) * 0.1)
        self.in_proj_bias = torch.nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, query, key=None, value=None, **kwargs):
        qkv = torch.nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        weights = torch.softmax((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5, dim=-1)
        return self.out_proj(weights @ v)


class _BrMLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = torch.nn.Linear(dim, dim)
        self.c_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.c_proj(torch.relu(self.c_fc(x)))


class _BrBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(dim)
        self.attn = _BrAttn(dim)
        self.ln_2 = torch.nn.LayerNorm(dim)
        self.mlp = _BrMLP(dim)

    def forward(self, x, attn_mask=None, **kwargs):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _BrVisual(torch.nn.Module):
    def __init__(self, in_dim=6, width=8, depth=3):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_BrBlock(width) for _ in range(depth)])
        self.ln_post = torch.nn.LayerNorm(width)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, 4, 1)
        x = self.input_proj(x)
        for block in self.transformer.resblocks:
            x = block(x)
        return self.ln_post(x).mean(dim=1)


class _BrModel(torch.nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.visual = _BrVisual(depth=depth)

    def encode_image(self, x):
        return self.visual(x)


def _brace_vision_models(depth=3):
    torch.manual_seed(0)
    base = _BrModel(depth)
    ft = deepcopy(base)
    with torch.no_grad():
        for p in ft.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return base, ft


def _hash_brace_pair(base, ft, layout=None):
    base_sd, ft_sd = _visual_state(base), _visual_state(ft)
    task_vector = {k: ft_sd[k] - v for k, v in base_sd.items() if v.is_floating_point()}
    out = {
        "base_state": hash_tensor_dict(base_sd),
        "ft_state": hash_tensor_dict(ft_sd),
        "task_vector": hash_tensor_dict(task_vector),
    }
    if layout is not None:
        out["layout"] = hash_json(layout)
    return out


BRACE_VISION_CASES = {
    # case: (strategy, lmc_mode, source_depth, target_depth)
    "extend_duplicate_independent": ("duplicate_per_weight", "independent", 3, 5),
    "extend_duplicate_shared": ("duplicate_per_weight", "shared", 3, 5),
    "extend_duplicate_shared_ft": ("duplicate_per_weight", "shared_ft", 3, 5),
    "extend_interpolate_independent": ("interpolate_per_weight", "independent", 3, 5),
    "shrink_interpolate_independent": ("interpolate_per_weight", "independent", 3, 2),
    "shrink_interpolate_shared": ("interpolate_per_weight", "shared", 3, 2),
    "shrink_interpolate_shared_ft": ("interpolate_per_weight", "shared_ft", 3, 2),
}


@pytest.mark.parametrize("case", sorted(BRACE_VISION_CASES))
def test_brace_block_extender_vision(case):
    from merge_and_rebase.rebase.block_extension.config import BlockExtensionConfig
    from merge_and_rebase.rebase.block_extension.vision import run_block_extension

    strategy, lmc_mode, source_depth, target_depth = BRACE_VISION_CASES[case]
    base, ft = _brace_vision_models(source_depth)
    loader = _class_loader(n=16, in_dim=6, batch_size=4, seed=3)
    config = BlockExtensionConfig(
        target_layers_total=target_depth,
        insertion_order="bottom-top",
        extension_density="spread",
        extension_strategy=strategy,
        dampening_factor=1.0,
        n_batches_act=2,
        skip_correction=False,
        skip_final_ln=False,
        ridge_identity=1.0,
        ridge_weight=1e-6,
        lmc_mode=lmc_mode,
        verbose=False,
        show_progress=False,
    )
    layout: dict = {}
    final_depth = run_block_extension(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=loader,
        target_layers_total=target_depth,
        config=config,
        device="cpu",
        layout_out=layout,
    )
    assert final_depth == target_depth
    for part, digest in _hash_brace_pair(base, ft, layout).items():
        _check(f"brace_vision_{case}:{part}", digest)


@pytest.mark.parametrize(("direction", "source_depth", "target_depth"), [("extend", 3, 5), ("shrink", 3, 2)])
def test_brace_block_extender_vision_class_api(direction, source_depth, target_depth):
    from merge_and_rebase.rebase.block_extension.vision import BlockExtender

    base, ft = _brace_vision_models(source_depth)
    loader = _class_loader(n=16, in_dim=6, batch_size=4, seed=3)
    extender = BlockExtender(base, ft, "cpu", verbose=False, show_progress=False)
    final_depth = extender.extend_and_calibrate(
        loader=loader,
        n_batches=2,
        strategy="interpolate_per_weight",
        dampening_factor=1.0,
        blocks_to_add=None,
        target_layers_total=target_depth,
        insertion_order="bottom-top",
        extension_density="spread",
        skip_correction=False,
        skip_final_ln=False,
        ridge_identity=100.0,
        ridge_weight=1e-6,
        lmc_mode="independent",
    )
    assert final_depth == target_depth
    for part, digest in _hash_brace_pair(base, ft, extender.extension_layout).items():
        _check(f"brace_vision_class_api:{direction}:{part}", digest)


# --------------------------------------------------------------------------------------
# LLM: real (tiny, randomly initialised, config-only => offline) transformers Qwen2
# decoders through the real Qwen2 family adapter. BRACE (DecoderBlockExtender via
# run_block_extension_llm), then llm_rebase's own helpers and the exact transport calls
# its main() issues for theseus / theseus_gqa / bico.
# --------------------------------------------------------------------------------------

_LLM_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "premise one entails hypothesis two",
    "a short sentence",
    "numbers and words 1 2 3 and more words follow here",
    "golden hashes pin numerical behaviour exactly",
    "tiny decoder calibration text number six",
]


class _WordTokenizer:
    """Deterministic stand-in tokenizer (no vocab files): word -> stable id in [1, 120]."""

    def _ids(self, prompt, max_length):
        ids = [(sum(map(ord, w)) * 7 + len(w)) % 120 + 1 for w in prompt.split()][:max_length]
        return ids, [1] * len(ids)

    def __call__(self, prompts, *, truncation, max_length, padding):
        ids_rows, mask_rows = [], []
        for prompt in prompts:
            ids, mask = self._ids(prompt, max_length)
            pad = max_length - len(ids)
            ids_rows.append(ids + [0] * pad)
            mask_rows.append(mask + [0] * pad)
        return {"input_ids": ids_rows, "attention_mask": mask_rows}

    def pad(self, features, *, return_tensors, padding, max_length):
        return {k: torch.tensor([row[k][:max_length] for row in features]) for k in features[0]}


def _llm_loader():
    from merge_and_rebase.eval.llm_rebase import _build_text_calibration_loader

    return _build_text_calibration_loader(tokenizer=_WordTokenizer(), texts=_LLM_TEXTS, batch_size=2, max_length=12)


def _tiny_qwen2(*, hidden, layers, heads, kv_heads, inter, seed):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=inter,
        vocab_size=128,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


# Source: 32 hidden / 4 heads / 2 kv heads. Target: 48 hidden / 6 heads / 2 kv heads
# (head_dim 8 on both, GQA ratio 2 -> 3, so theseus_gqa's head assignment is non-trivial).
_LLM_SRC = dict(hidden=32, heads=4, kv_heads=2, inter=64)
_LLM_TGT = dict(hidden=48, heads=6, kv_heads=2, inter=96)
_LLM_DEPTHS = {"extend": (2, 4), "shrink": (4, 2)}


def _llm_source_pair(layers):
    base = _tiny_qwen2(layers=layers, seed=21, **_LLM_SRC)
    ft = deepcopy(base)
    torch.manual_seed(22)
    with torch.no_grad():
        for p in ft.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return base, ft


def _brace_llm_config(target_layers_total, *, lmc_mode="independent", skip_correction=False):
    from merge_and_rebase.rebase.block_extension.config import BlockExtensionConfig

    return BlockExtensionConfig(
        extension_strategy="interpolate_per_weight",
        target_layers_total=target_layers_total,
        lmc_mode=lmc_mode,
        skip_correction=skip_correction,
        n_batches_act=2,
        verbose=False,
        show_progress=False,
    )


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize("lmc_mode", ["independent", "shared"])
def test_brace_decoder_block_extender(lmc_mode, direction):
    from merge_and_rebase.rebase.block_extension.decoder import run_block_extension_llm
    from merge_and_rebase.rebase.model_families import infer_family

    source_depth, target_depth = _LLM_DEPTHS[direction]
    base, ft = _llm_source_pair(source_depth)
    layout: dict = {}
    final_depth = run_block_extension_llm(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=_llm_loader(),
        target_layers_total=target_depth,
        config=_brace_llm_config(target_depth, lmc_mode=lmc_mode),
        family_adapter=infer_family(base),
        device="cpu",
        layout_out=layout,
    )
    assert final_depth == target_depth
    for part, digest in _hash_brace_pair(base, ft, layout).items():
        _check(f"brace_decoder_{lmc_mode}:{direction}:{part}", digest)


@pytest.mark.parametrize("direction", ["extend", "shrink"])
def test_brace_decoder_block_extender_class_api(direction):
    from merge_and_rebase.rebase.block_extension.decoder import DecoderBlockExtender
    from merge_and_rebase.rebase.model_families import infer_family

    source_depth, target_depth = _LLM_DEPTHS[direction]
    base, ft = _llm_source_pair(source_depth)
    extender = DecoderBlockExtender(base, ft, infer_family(base), device="cpu", verbose=False, show_progress=False)
    final_depth = extender.extend_and_calibrate(
        loader=_llm_loader(),
        n_batches=2,
        strategy="interpolate_per_weight",
        target_layers_total=target_depth,
        insertion_order="bottom-top",
        extension_density="spread",
        skip_correction=False,
    )
    assert final_depth == target_depth
    for part, digest in _hash_brace_pair(base, ft).items():
        _check(f"brace_decoder_class_api:{direction}:{part}", digest)


def _llm_orchestration(method_name, direction):
    """llm_rebase.main()'s per-task flow minus checkpoint/tokenizer IO."""
    import merge_and_rebase.rebase.methods  # noqa: F401
    from merge_and_rebase.eval.llm_rebase import _prepare_resized_task_delta
    from merge_and_rebase.merge.runtime import to_cpu_fp32
    from merge_and_rebase.rebase.model_families import infer_family
    from merge_and_rebase.rebase.registry import get_method

    source_depth, target_depth = _LLM_DEPTHS[direction]
    base, ft = _llm_source_pair(source_depth)
    target = _tiny_qwen2(layers=target_depth, seed=23, **_LLM_TGT)
    adapter = infer_family(base)
    prepared_task = _prepare_resized_task_delta(
        source_base_model=base,
        source_ft_model=ft,
        calibration_loader=_llm_loader(),
        target_layers_total=target_depth,
        config=_brace_llm_config(target_depth),
        family_adapter=adapter,
        device="cpu",
    )
    target_base_sd = to_cpu_fp32(target.state_dict())
    body_delta = {k: v for k, v in prepared_task.delta.items() if k in prepared_task.transport_keys}
    method = get_method(method_name)
    shared = dict(
        source_model=prepared_task.source_model,
        target_model=target,
        source_dataloader=_llm_loader(),
        target_dataloader=_llm_loader(),
        family_adapter=adapter,
        device="cpu",
    )
    kwargs = {"seq_align": "interpolate", "n_batches": 2, "verbose": False, "show_progress": False}
    if method_name.startswith("bico"):
        from merge_and_rebase.models.grad_recipes import causal_lm_recipe

        shared["source_recipe"] = causal_lm_recipe(device="cpu")
        shared["target_recipe"] = causal_lm_recipe(device="cpu")
        kwargs["curvature_dataloader"] = None
    transported = method.transport(
        source_base=prepared_task.source_base,
        target_base=target_base_sd,
        delta=body_delta,
        strict=False,
        prepared=None,
        **shared,
        **kwargs,
    )
    assert transported
    assert any(float(t.abs().sum()) > 0.0 for t in transported.values()), "transport produced only zeros"
    summary = {
        "resized_source_base": hash_tensor_dict(prepared_task.source_base),
        "corrected_delta": hash_tensor_dict(prepared_task.delta),
        "uncorrected_delta": hash_tensor_dict(prepared_task.uncorrected_delta),
        "layout": prepared_task.extension_layout,
        "transport_keys": sorted(prepared_task.transport_keys),
    }
    return transported, summary


@pytest.mark.parametrize("direction", ["extend", "shrink"])
@pytest.mark.parametrize("method_name", ["theseus", "theseus_gqa", "bico"])
def test_llm_rebase_orchestration(method_name, direction):
    transported, summary = _llm_orchestration(method_name, direction)
    _check(f"llm_rebase_{method_name}:{direction}:brace_and_delta", hash_json(summary))
    _check(f"llm_rebase_{method_name}:{direction}:transported_delta", hash_tensor_dict(transported))
