- Base Models
    - Vision Models
        [x] ViT-B-32 openai
        [] ViT-B-16 openai
        [] ViT-L-14 openai
    - Language Models
        [] T5
        [] Llama
        [] WizardLM-13
        [] WizardMath-13B
- Benchmarks
    [x] 8-Vision
    [x] 14-Vision
    [x] 20-Vision
    [] 5-GLUE (MNLI SNLI QNLI SICK RTE SCITAIL) knots
    [] 4-GLUE (MRPC RTE CoLA SST-2) TA
- Merging
    - Methods
        [] Simple Averaging
        [x] Weighted Averaging
        [] RegMean
        [] Consensus TA
        [] Fisher Merging
        [x] Task Arithmetic
        [x] Ties Merging
        [x] DARE
        [] DARE TIES
        [x] TSVM
        [x] ISO-C
        [x] ISO-CTS
        [x] CART
        [] PCB
        [] MaTS
    - Post Merging Methods
        [] LiNeS
        [] Subspace Boosting
    - Quality of Life
        [x] Single Accuracy Evaluation
        [x] Multi Accuracy Evaluation
        [x] Alpha Search with Early Stopping
        [] Multi Parameter Search with Early Stopping
        [x] Caching merged model
    - Merging LoRA
        [x] Full Space
        [x] KnOTS
        [x] Core Space
    - Routing Settings
        [] Fixed Routing
            [] Mass MoErging
            [] TSVC
            [] Twin Merging
            [] SMILE
            [] TALL Mask
        [] Learned Routing
    - Analysis Tools
        [x] Loss Landscape Visualization
        [] SAR
- Fine-Tuning
    [x] Full Single Fine-Tuning
    [] Full Joint Fine-Tuning
    [] Full Sequential Fine-Tuning
    [] PeFT
        [x] LoRA
        [] VeRA
        [] IA3
    - Linearized Fine-Tuning
        [x] NTK
        [] KFAC
- Rebase Methods
    [] Git-Rebasin
    [] TransFusion
    [] GradFix
    [] Theseus
