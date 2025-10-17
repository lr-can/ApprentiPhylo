import os
import shutil

# Define the new base directory
new_base_dir = "results/MPD/viridiplantae_group_results"
os.makedirs(new_base_dir, exist_ok=True)

# Define groups and subgroups
groups = {
    "group1_four_model_F": [
        "DSO78_frequencies",
        "JTT92_frequencies",
        "LG08_frequencies",
        "WAG_frequencies"
    ],
    "group2_WAG_posterior_extra_length": {
        "with_data2": [
            "WAG_frequencies_posterior_extra_length_data2_ext_0",
            "WAG_frequencies_posterior_extra_length_data2_ext_0.05",
            "WAG_frequencies_posterior_extra_length_data2_ext_0.1",
            "WAG_frequencies_posterior_extra_length_data2_ext_0.2",
            "WAG_frequencies_posterior_extra_length_data2_ext_0.5"
        ],
        "without_data2": [
            "WAG_frequencies_posterior_extra_length_ext_0",
            "WAG_frequencies_posterior_extra_length_ext_0.05",
            "WAG_frequencies_posterior_extra_length_ext_0.1",
            "WAG_frequencies_posterior_extra_length_ext_0.2",
            "WAG_frequencies_posterior_extra_length_ext_0.5"
        ]
    },
    "group3_WAG_sampling_seq_root": {
        "with_data2": [
            "WAG_sampling_seq_data2_root_0",
            "WAG_sampling_seq_data2_root_0.05",
            "WAG_sampling_seq_data2_root_0.1",
            "WAG_sampling_seq_data2_root_0.2",
            "WAG_sampling_seq_data2_root_0.5"
        ],
        "without_data2": [
            "WAG_sampling_seq_root_0",
            "WAG_sampling_seq_root_0.05",
            "WAG_sampling_seq_root_0.1",
            "WAG_sampling_seq_root_0.2",
            "WAG_sampling_seq_root_0.5"
        ]
    },
    "group4_WAG_basic_comparison": {
        "with_data2": [
            "WAG_frequencies_posterior_extra_length_data2_ext_0",
            "WAG_sampling_seq_data2_root_0",
            "WAG_frequencies_sampling_seq_data2"
        ],
        "without_data2": [
            "WAG_frequencies",
            "WAG_frequencies_posterior_extra_length_ext_0",
            "WAG_sampling_seq_root_0",
            "WAG_frequencies_sampling_seq"
        ]
    }
}

# Iterate through groups and subgroups, create folders and copy files
for group, subgroups in groups.items():
    group_dir = os.path.join(new_base_dir, group)
    os.makedirs(group_dir, exist_ok=True)
    
    if isinstance(subgroups, list):
        # Copy files directly (e.g., group1)
        for model in subgroups:
            src_dir = os.path.join("results/MPD/viridiplantae_results", model)
            dst_dir = os.path.join(group_dir, model)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    else:
        # Process subgroups (e.g., group2, group3, group4)
        for subgroup, models in subgroups.items():
            subgroup_dir = os.path.join(group_dir, subgroup)
            os.makedirs(subgroup_dir, exist_ok=True)
            for model in models:
                src_dir = os.path.join("results/MPD/viridiplantae_results", model)
                dst_dir = os.path.join(subgroup_dir, model)
                if os.path.exists(src_dir):
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
