"""Synthetic source admission report for isolated verifier controls only."""
import remote
def controls_fixture():
    rows = []
    for method in ('direct_to_fixture', 'module_to_fixture', 'direct_quantize_meta', 'module_apply_meta'):
        rows.append(dict(method=method, scope='CPU_META_TYPE_MECHANICS_ONLY', identity=True,
                         **{'class': True}, attrs=True, module_alias=True,
                         bias_device='meta' if method == 'module_apply_meta' else 'cpu'))
    rows.extend([{'failure': 'ORIGINAL_INCOMPATIBLE_TYPE', 'original_unchanged': True} for _ in range(3)])
    for negative in ('python_weakref', 'cpp_weakref', 'held_impl', 'requires_grad', 'subclass'):
        rows.append(dict(negative=negative, rejected=True, tensorimpl_unchanged=True,
                         dict_identity_unchanged=True, data_unchanged=True, module_unchanged=True, error='SYNTHETIC rejection'))
    rows.append(dict(compatible_cpu=True, weight_identity=True, custom_attrs=True,
                     state_format=True, weights_only_load=True, keys=['bias', 'weight']))
    for case in ('existing_gradient', 'overwrite_flag', 'swap_flag'):
        flags = ('rejected_before_swap', 'parameter_identity', 'tensorimpl_identity', 'dict_identity',
                 'class_identity', 'values_unchanged', 'quant_state_identity', 'module_alias_unchanged', 'gradient_identity', 'bias_identity')
        rows.append(dict(case=case, scope='CPU_META_TYPE_MECHANICS_ONLY', **{flag: True for flag in flags}))
    rows.append(dict(case='genuine_cpu_forward_backward', forward_exact=True, dx_exact=True, db_exact=True,
                     frozen_base=True, state_plain_tensors=True, forward_values=[[0.0, 0.0]] * 3))
    rows.append(dict(case='fresh_process_cpu_public_restore', forward_exact=True, weights_only=True,
                     original_classes=True, module_state_alias=True, frozen_base=True))
    return {'record_validation': 'PASS', 'qualified_source_controls': True,
            'runtime': {'torch': '2.9.0+cpu', 'python': '3.12.14', 'platform': 'Linux'},
            'patch_manifest_sha256': remote.TRANSFER_PATCH_MANIFEST_SHA, 'tpu': 'NOT_RUN', 'xla': 'NOT_RUN',
            'whole_module_transactionality': 'NOT_PROMISED', 'failures': [], 'controls': rows}
