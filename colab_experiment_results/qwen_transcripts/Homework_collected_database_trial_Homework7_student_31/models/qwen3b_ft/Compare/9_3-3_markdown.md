![Source Image](9_3-3_source.png)
 

KCL:

\( i_r - i_{R_2} + i_L + (10 \times 10^{-6}) \frac{dW}{dt} = 0 \)

KVL:

\( W = 2i_L + 10^{-3} \frac{di_c}{dt} \)

Plug \( W \) into KCL

\( 2i_L + 10^{-3} \frac{di_c}{dt} - i_{R_2} + i_C + (10 \times 10^{-6}) \left[ 2 \frac{di_c}{dt} + 10^{-3} \frac{di}{dt^2} \right] \)

\( i_{R_2} = 3i_L + 0.00102 \frac{di_c}{dt} + (1 \times 10^{-8}) \frac{di^2}{dt^2} \)

\( \frac{di_c}{dt^2} + 102000 \frac{di_c}{dt} + (3 \times 10^{-8}) i_L = (1 \times 10^2) i_{R_2} \)

\(\Rightarrow \)

\( 3^2 + 102000 \cdot 2 + 3 \times 10^9 = 0 \)

Roots:

\(- \frac{102000 \pm \sqrt{(102000)^2 - 4(3)(3 \times 10^9)}}{2} \Rightarrow \)

\[
\begin{cases}
i_1 = -3031 \\
i_2 = -98969
\end{cases}
\]

