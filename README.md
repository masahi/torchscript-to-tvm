For a background on motivation of this repo, see [this post](https://discuss.tvm.ai/t/discuss-adding-a-pytorch-frontend/5026/6) on TVM discussion forum.


This is a demo of translating PyTorch modules with control flow to TVM Relay via TorchScript.


* TorchScript `prim::If` node maps directly to Relay's `If` expression.
* `prim::Loop` node is translated to conditional and tail recursion in Relay.

Below is an example of how a simple TorchScript module looks like after translation to PyTorch JIT IR and TVM Relay.
See dynamic_test.py for more examples.


PyTorch module
```Python
class LoopWithIf(torch.nn.Module):
    def forward(self, inp):
        a = inp
        for i in range(inp.size(0)):
            b = a * 2
            b = a + b
            if b.sum() > 0.0:
                a += b
            else:
                a -= b
        return a
```


PyTorch JIT IR

```
graph(%self : __torch__.LoopWithIf,
      %inp.1 : Tensor):
  %2 : None = prim::Constant()
  %3 : int = prim::Constant[value=1]()
  %4 : bool = prim::Constant[value=1]() # dynamic_test.py:64:8
  %5 : int = prim::Constant[value=0]() # dynamic_test.py:64:32
  %6 : int = prim::Constant[value=2]() # dynamic_test.py:65:20
  %7 : float = prim::Constant[value=0]() # dynamic_test.py:67:25
  %8 : int = aten::size(%inp.1, %5) # dynamic_test.py:64:23
  %a : Tensor = prim::Loop(%8, %4, %inp.1) # dynamic_test.py:64:8
    block0(%i : int, %a.15 : Tensor):
      %b.1 : Tensor = aten::mul(%a.15, %6) # dynamic_test.py:65:16
      %b.3 : Tensor = aten::add(%a.15, %b.1, %3) # dynamic_test.py:66:16
      %14 : Tensor = aten::sum(%b.3, %2) # dynamic_test.py:67:15
      %15 : Tensor = aten::gt(%14, %7) # dynamic_test.py:67:15
      %16 : bool = aten::Bool(%15) # dynamic_test.py:67:15
      %a.14 : Tensor = prim::If(%16) # dynamic_test.py:67:12
        block0():
          %a.4 : Tensor = aten::add_(%a.15, %b.3, %3) # dynamic_test.py:68:16
          -> (%a.4)
        block1():
          %a.7 : Tensor = aten::sub_(%a.15, %b.3, %3) # dynamic_test.py:70:16
          -> (%a.7)
      -> (%4, %a.14)
  return (%a)
```


TVM Relay IR
```
v0.0.4
def @main(%X: Tensor[(10, 20), float32]) -> Tensor[(10, 20), float32] {
  %9 = (
    let %while_loop: fn (int32, Tensor[(10, 20), float32]) -> (int32, Tensor[(10, 20), float32]) = fn (%i: int32, %a.15: Tensor[(10, 20), float32]) -> (int32, Tensor[(10, 20), float32]) {
      %0 = greater_equal(%i, 1 /* ty=int32 */) /* ty=bool */;
      %1 = less_equal(%i, 10 /* ty=int32 */) /* ty=bool */;
      %2 = logical_and(%0, %1) /* ty=bool */;
      if (%2) {
        %3 = add(%i, 1 /* ty=int32 */) /* ty=int32 */;
        %4 = multiply(%a.15, 2f /* ty=float32 */) /* ty=Tensor[(10, 20), float32] */;
        %5 = add(%a.15, %4) /* ty=Tensor[(10, 20), float32] */;
        %6 = sum(%5) /* ty=float32 */;
        %7 = greater(%6, 0f /* ty=float32 */) /* ty=bool */;
        %8 = if (%7) {
          add(%a.15, %5) /* ty=Tensor[(10, 20), float32] */
        } else {
          subtract(%a.15, %5) /* ty=Tensor[(10, 20), float32] */
        };
        %while_loop(%3, %8) /* ty=(int32, Tensor[(10, 20), float32]) */
      } else {
        (%i, %a.15)
      }
    };
    %while_loop
  );
  %10 = %9(1 /* ty=int32 */, %X) /* ty=(int32, Tensor[(10, 20), float32]) */;
  %10.1
}
```
