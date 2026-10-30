![Source Image](9_3-3_source.png)
 

KVL to the right: \( 2i_L + L \frac{d i_L}{dt} - v_c = 0 \quad \textcolor{red}{②} \)

Operator method:
\[
\begin{cases}
v_s - v_c - i_L - L s v_c = 0 \\
2i_L + L s i_L - v_c = 0 \quad \Rightarrow \quad i_L = \frac{v_c}{2 + L s}
\end{cases}
\]

Plug into \(\textcolor{red}{①}\):
\[
v_s - v_c = \frac{v_c}{2 + L s} - L s v_c = 0 \quad \Rightarrow \quad (2 + L s) v_c + (2 + L s) v_c + v_c = (2 + L s) v_c \quad \Rightarrow \quad (2 + L s) (L s v_c + (2 + L s) v_c + 3 v_c) = 0
\]
\[
\Rightarrow L s^2 v_c + 1.02 \cdot 10^3 \cdot s \cdot v_c + 3 \cdot 10^8 v_c = 0
\]

Characteristic equation:
\[
s^2 + 1.02 \cdot 10^3 \cdot s + 3 \cdot 10^8 = 0
\]

Roots:
\[
s_1 = -3,031,126, \quad s_2 = -989,687.7
\]

