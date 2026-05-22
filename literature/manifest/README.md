# Literature seed corpus

Updated: 2026-05-21T03:40:57

- Total records: 25
- Downloaded PDFs: 20
- Metadata-only / not downloaded: 5

This corpus includes the NeuroPilot manuscript and supplementary materials together with a seed set of related calcium imaging, behavior, and analysis-method papers. Downloads use openly accessible publisher PDFs, public author manuscripts, institutional repositories, PMC/Journal public PDFs, arXiv/eLife PDFs, or public lab-hosted PDFs. Paywalled PDFs were not bypassed.

## Neuropilot
- `downloaded` neuropilot_main_20260420: NeuroPilot main manuscript (2026, manuscript)
  - path/source: literature/raw/Neuropilot/NeuroPilot_main_20260420.pdf
- `downloaded` neuropilot_supplementary_materials_20260419: NeuroPilot supplementary materials (2026, manuscript)
  - path/source: literature/raw/Neuropilot/NeuroPilot_supplementary_materials_20260419.pdf

## anderson_lab
- `downloaded` lee_esr1_vmhvl_2014_nature: Scalable control of mounting and attack by Esr1+ neurons in the ventromedial hypothalamus (2014, Nature)
  - path/source: literature/raw/anderson_lab/lee_esr1_vmhvl_2014_nature__scalable_control_of_mounting_and_attack_by_esr1_neurons_in_the_ventromedial_hypothalamus.pdf
  - note: Accepted manuscript from CaltechAUTHORS.
- `downloaded` hong_social_behavior_2015_pnas: Automated measurement of mouse social behaviors using depth sensing, video tracking, and machine learning (2015, PNAS)
  - path/source: literature/raw/anderson_lab/hong_social_behavior_2015_pnas__automated_measurement_of_mouse_social_behaviors_using_depth_sensing_video_tracki.pdf
- `downloaded` remedios_social_sex_2017_nature: Social behaviour shapes hypothalamic neural ensemble representations of conspecific sex (2017, Nature)
  - path/source: literature/raw/anderson_lab/remedios_social_sex_2017_nature__social_behaviour_shapes_hypothalamic_neural_ensemble_representations_of_conspeci.pdf
- `metadata_only` karigo_mounting_2021_nature: Distinct hypothalamic control of same- and opposite-sex mounting behaviour in mice (2021, Nature)
  - path/source: https://doi.org/10.1038/s41586-020-2995-0
  - note: CaltechAUTHORS exposes supplementary files but not the main article PDF; PMC PDF endpoint is protected and PMC OA API reports not open access.
- `downloaded` yang_social_network_2022_nature: Transformations of neural representations in a social behaviour network (2022, Nature)
  - path/source: literature/raw/anderson_lab/yang_social_network_2022_nature__transformations_of_neural_representations_in_a_social_behaviour_network.pdf
- `downloaded` line_attractor_aggression_2023_cell: An approximate line attractor in the hypothalamus encodes an aggressive state (2023, Cell)
  - path/source: literature/raw/anderson_lab/line_attractor_aggression_2023_cell__an_approximate_line_attractor_in_the_hypothalamus_encodes_an_aggressive_state.pdf
  - note: Publisher PDF available through CaltechAUTHORS.

## behavior_datasets
- `downloaded` computational_neuroethology_2019_neuron: Computational Neuroethology: A Call to Action (2019, Neuron)
  - path/source: literature/raw/behavior_datasets/computational_neuroethology_2019_neuron__computational_neuroethology_a_call_to_action.pdf
  - note: Author-posted PDF from Datta lab site. TLS hostname mismatch required certificate verification bypass; URL retained.
- `downloaded` calms21_2021_arxiv: The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions (2021, NeurIPS Datasets and Benchmarks / arXiv)
  - path/source: literature/raw/behavior_datasets/calms21_2021_arxiv__the_multi_agent_behavior_dataset_mouse_dyadic_social_interactions.pdf
  - note: arXiv PDF downloaded; CaltechAUTHORS mirror retained in urls.

## behavior_neural_dynamics
- `downloaded` single_trial_movements_2019_nat_neuro: Single-trial neural dynamics are dominated by richly varied movements (2019, Nature Neuroscience)
  - path/source: literature/raw/behavior_neural_dynamics/single_trial_movements_2019_nat_neuro__single_trial_neural_dynamics_are_dominated_by_richly_varied_movements.pdf
  - note: Author manuscript PDF from eScholarship.

## methods
- `metadata_only` cellreg_2017_cell_reports: Tracking the Same Neurons across Multiple Days in Ca2+ Imaging Data (2017, Cell Reports)
  - path/source: https://www.sciencedirect.com/science/article/pii/S2211124717314304
  - note: Cell Reports article is open access, but Cell/ScienceDirect and Weizmann PDF endpoints returned bot/403 responses in this environment. Keep DOI/landing page for later manual or browser-assisted download.
- `downloaded` normcorre_2017_jneumeth: NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data (2017, Journal of Neuroscience Methods)
  - path/source: literature/raw/methods/normcorre_2017_jneumeth__normcorre_an_online_algorithm_for_piecewise_rigid_motion_correction_of_calcium_imaging_data.pdf
  - note: bioRxiv v2 public PDF mirrored by ResearchHub; journal landing page retained as landing_page.
- `downloaded` cnmfe_2018_elife: Efficient and accurate extraction of in vivo calcium signals from microendoscopic video data (2018, eLife)
  - path/source: literature/raw/methods/cnmfe_2018_elife__efficient_and_accurate_extraction_of_in_vivo_calcium_signals_from_microendoscopi.pdf
- `downloaded` min1pipe_2018_cell_reports: MIN1PIPE: A Miniscope 1-Photon-Based Calcium Imaging Signal Extraction Pipeline (2018, Cell Reports)
  - path/source: literature/raw/methods/min1pipe_2018_cell_reports__min1pipe_a_miniscope_1_photon_based_calcium_imaging_signal_extraction_pipeline.pdf
  - note: Open-access Cell Reports PDF mirrored by Frohlich lab.
- `downloaded` caiman_2019_elife: CaImAn an open source tool for scalable calcium imaging data analysis (2019, eLife)
  - path/source: literature/raw/methods/caiman_2019_elife__caiman_an_open_source_tool_for_scalable_calcium_imaging_data_analysis.pdf
- `metadata_only` deepcad_2021_nat_methods: Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising (2021, Nature Methods)
  - path/source: https://doi.org/10.1038/s41592-021-01225-0
  - note: Nature article page is accessible, but direct PDF is not openly retrievable in this environment; bioRxiv endpoint is protected by an anti-bot challenge. Keep DOI/landing page for later manual or API-based enrichment.
- `metadata_only` microendoscopy_review_2021_jneumeth: Fluorescence microendoscopy for in vivo deep-brain imaging of neuronal circuits (2021, Journal of Neuroscience Methods)
  - path/source: https://doi.org/10.1016/j.jneumeth.2020.109015
  - note: PMC full-text HTML is accessible, but the PDF endpoint is protected and the article is not in the PMC OA package API. Keep PMC landing page for text/abstract ingestion later.
- `downloaded` minian_2022_elife: Minian, an open-source miniscope analysis pipeline (2022, eLife)
  - path/source: literature/raw/methods/minian_2022_elife__minian_an_open_source_miniscope_analysis_pipeline.pdf
  - note: eLife CDN CC-BY article PDF.

## stringer_pachitariu
- `downloaded` suite2p_2017_biorxiv: Suite2p: beyond 10,000 neurons with standard two-photon microscopy (2017, bioRxiv / UCL Discovery)
  - path/source: literature/raw/stringer_pachitariu/suite2p_2017_biorxiv__suite2p_beyond_10_000_neurons_with_standard_two_photon_microscopy.pdf
- `downloaded` spike_deconv_2018_jneurosci: Robustness of Spike Deconvolution for Neuronal Calcium Imaging (2018, Journal of Neuroscience)
  - path/source: literature/raw/stringer_pachitariu/spike_deconv_2018_jneurosci__robustness_of_spike_deconvolution_for_neuronal_calcium_imaging.pdf
  - note: Journal of Neuroscience public PDF.
- `downloaded` calcium_processing_review_2019_conb: Computational processing of neural recordings from calcium imaging data (2019, Current Opinion in Neurobiology)
  - path/source: literature/raw/stringer_pachitariu/calcium_processing_review_2019_conb__computational_processing_of_neural_recordings_from_calcium_imaging_data.pdf
- `downloaded` high_dim_geometry_2019_nature: High-dimensional geometry of population responses in visual cortex (2019, Nature)
  - path/source: literature/raw/stringer_pachitariu/high_dim_geometry_2019_nature__high_dimensional_geometry_of_population_responses_in_visual_cortex.pdf
  - note: Author/preprint PDF from UCL Discovery.
- `downloaded` spontaneous_behaviors_2019_science: Spontaneous behaviors drive multidimensional, brain-wide activity (2019, Science)
  - path/source: literature/raw/stringer_pachitariu/spontaneous_behaviors_2019_science__spontaneous_behaviors_drive_multidimensional_brain_wide_activity.pdf
- `metadata_only` cellpose_2021_nat_methods: Cellpose: a generalist algorithm for cellular segmentation (2021, Nature Methods)
  - path/source: https://doi.org/10.1038/s41592-020-01018-x
  - note: Nature PDF redirects to HTML; bioRxiv PDF endpoint returns anti-bot challenge. Keep DOI/landing page for later manual or institutional download.
