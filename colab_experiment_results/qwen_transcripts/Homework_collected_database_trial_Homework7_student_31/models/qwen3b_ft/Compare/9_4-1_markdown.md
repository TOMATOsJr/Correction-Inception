![Source Image](9_4-1_source.png)
 

Using operators:

\[
(LC s^2 + \frac{1}{R} s + 1) v = N_2 \quad \Rightarrow \quad s^2 + \frac{1}{RC} s + \frac{1}{LC} = 0
\]

Roots: 
\[
-\frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \quad \Rightarrow \quad s_1 = -100 \quad \text{and} \quad s_2 = -400
\]

\(v(t) = Ae^{-100t} + Be^{-400t}\)

\(v(0) = 6 = A + B \quad \Rightarrow \quad A + B = 6\)

\(\frac{dv}{dt}(0) = -3000 = -100A + 400B \quad \Rightarrow \quad -100A - 400B = -3000\)

\(-100(6 - B) - 400B = -3000 \quad \Rightarrow \quad -600 + 100B - 400B = -3000\)

\(-300B = -2400 \quad \Rightarrow \quad B = 8\)

\(A = 6 - B = 6 - 8 = -2\)

\(\therefore \quad v(t) = -2e^{-100t} + 8e^{-400t} \quad \text{for} \quad t > 0\)

