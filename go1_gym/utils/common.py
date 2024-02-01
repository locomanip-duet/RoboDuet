from yapf.yapflib.yapf_api import FormatCode
import torch
import random
import numpy as np
import os
import sys
import select


def input_with_timeout(timeout):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        s = sys.stdin.readline()
        try:
            return s.strip()
        except:
            return None
    else:
        return None


def format_code(code_text: str):
    """Format the code text with yapf."""
    yapf_style = dict(
        based_on_style='pep8',
        blank_line_before_nested_class_or_def=True,
        split_before_expression_after_opening_paren=True)
    try:
        code_text, _ = FormatCode(code_text, style_config=yapf_style)
    except:  # noqa: E722
        raise SyntaxError('Failed to format the config file, please '
                          f'check the syntax of: \n{code_text}')

    return code_text

def quaternion_to_rpy(quaternions):
    """
    Note:
        rpy (torch.Tensor): Tensor of shape (N, 3). Range: (-pi, pi)
    """
    assert quaternions.shape[1] == 4, "Input should have shape (N, 4)"
    
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    rpy = torch.zeros((quaternions.shape[0], 3), device=quaternions.device, dtype=quaternions.dtype)
    
    # Compute Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    rpy[:, 0] = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Compute Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    rpy[:, 1] = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.tensor(torch.pi/2, device=quaternions.device, dtype=quaternions.dtype), torch.asin(sinp))
    
    # Compute Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    rpy[:, 2] = torch.atan2(siny_cosp, cosy_cosp)
    
    return rpy

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed