"""Explicit Tensor shifts for diagnostic variants only. No plugin replacement."""
import torch

def tensor_shift(left,right,direction):
 if type(left)is int and type(right)is int:
  return left<<right if direction=='left'else left>>right
 if not isinstance(left,torch.Tensor)or left.dtype!=torch.int32:
  raise ValueError('DIAGNOSTIC_SHIFT_LEFT_INT32_TENSOR_REQUIRED')
 if type(right)is int:right=torch.tensor(right,dtype=torch.int32,device=left.device)
 if not isinstance(right,torch.Tensor)or right.dtype!=torch.int32 or right.device!=left.device:
  raise ValueError('DIAGNOSTIC_SHIFT_COUNT_INT32_TENSOR_REQUIRED')
 if direction=='left':return torch.ops.aten.bitwise_left_shift.Tensor(left,right)
 if direction=='right':return torch.ops.aten.bitwise_right_shift.Tensor(left,right)
 raise ValueError('DIAGNOSTIC_SHIFT_DIRECTION')
