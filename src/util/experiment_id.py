def generate_experiment_id(
    name,
    debiasing_method=None,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
    source_language=None,
    target_language=None,
    pre_assignment_type=None,
):
    experiment_id = f"e-{name}"

    # Build the experiment ID.
    if isinstance(debiasing_method, str):
        experiment_id += f"_d-{debiasing_method}"
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        if 'meta-llama/' in model_name_or_path:
            new_model_name_or_path = model_name_or_path.split('/')[1]
        elif 'mistralai/' in model_name_or_path:
            new_model_name_or_path = model_name_or_path.split('/')[1]
        else:
            new_model_name_or_path = model_name_or_path
        experiment_id += f"_c-{new_model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_i-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(source_language, str):
        experiment_id += f"_a-{source_language}"
    if isinstance(target_language, str):
        experiment_id += f"_b-{target_language}"
    if isinstance(pre_assignment_type, str):
        experiment_id += f"_p-{pre_assignment_type}"

    return experiment_id
