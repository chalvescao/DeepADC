# Results 

## 9/4/2017

- Running code in onelayer
  - Two versions, one with and one without adding a derivative cost.
  - Train for 20k iterations. 10k with lr = 1e-2, 5k with 1e-2.5 and 5k with 1e-3
  - Add weight decay to all weights (seems to help stabilize).
  
- 8 Bit, 64 Hidden, Derivative Loss
<img src="9.4.ayan/b8h64.png" height=400px>

- 8 Bit, 64 Hidden, No Derivative Loss
<img src="9.4.ayan/b8h64ND.png" height=400px>

- 8 Bit, 32 Hidden, Derivative Loss
<img src="9.4.ayan/b8h32.png" height=400px>

- 8 Bit, 32 Hidden, No Derivative Loss
<img src="9.4.ayan/b8h32ND.png" height=400px>

- 6 Bit, 36 Hidden, Derivative Loss
<img src="9.4.ayan/b6h36.png" height=400px>

- 6 Bit, 36 Hidden, No Derivative Loss
<img src="9.4.ayan/b6h36ND.png" height=400px>

- 4 Bit, 16 Hidden, Derivative Loss
<img src="9.4.ayan/b4h16.png" height=400px>

- 4 Bit, 16 Hidden, No Derivative Loss
<img src="9.4.ayan/b4h16ND.png" height=400px>
