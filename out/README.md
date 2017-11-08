# Results 

## 9/8/2017

- Running new code in onelayer
  - Leave out derivative cost
  - Add different weighted versions of original 'classification' cost
  - Make the 'difference' cost be wrt un-quantized version
  - Conclusion: for fewer bits, helps to have higher weight on classification cost

### 8 Bits (64 hidden)

- Weight 0.5 on classification cost
<img src="9.8.ayan/b8h64_c.5.png" height=400px>
- Weight 1 on classification cost
<img src="9.8.ayan/b8h64_c1.png" height=400px>
- Weight 2 on classification cost
<img src="9.8.ayan/b8h64_c2.png" height=400px>
- Weight 4 on classification cost
<img src="9.8.ayan/b8h64_c4.png" height=400px>

### 6 Bits (36 hidden)

- Weight 0.5 on classification cost
<img src="9.8.ayan/b6h36_c.5.png" height=400px>
- Weight 1 on classification cost
<img src="9.8.ayan/b6h36_c1.png" height=400px>
- Weight 2 on classification cost
<img src="9.8.ayan/b6h36_c2.png" height=400px>
- Weight 4 on classification cost
<img src="9.8.ayan/b6h36_c4.png" height=400px>

### 4 Bits (16 hidden)

- Weight 0.5 on classification cost
<img src="9.8.ayan/b4h16_c.5.png" height=400px>
- Weight 1 on classification cost
<img src="9.8.ayan/b4h16_c1.png" height=400px>
- Weight 2 on classification cost
<img src="9.8.ayan/b4h16_c2.png" height=400px>
- Weight 4 on classification cost
<img src="9.8.ayan/b4h16_c4.png" height=400px>

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
