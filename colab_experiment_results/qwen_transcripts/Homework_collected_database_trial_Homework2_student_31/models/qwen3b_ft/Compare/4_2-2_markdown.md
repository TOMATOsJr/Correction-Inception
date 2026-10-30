![Source Image](4_2-2_source.png)
 

KCL @ Node \( V_1 \):
\[
\frac{v_1 - v_2}{20} + 2 = \frac{v_2 - v_3}{10}
\]

KCL @ Node \( V_2 \):
\[
\frac{v_2 - v_3}{10} + 1 = \frac{v_3}{15}
\]

KCL @ Node \( V_3 \):
\[
1 + \frac{(v_1 - v_2)}{20} + \frac{v_1}{5} = 0
\]

\[
\Rightarrow v_1 + 3v_2 - 2v_3 = 40
\]
\[
-3v_2 + 5v_3 = 30
\]
\[
5v_1 - v_2 = -20
\]

\[
\begin{bmatrix}
-1 & 3 & -2 \\
0 & -3 & 5 \\
5 & -1 & 0
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix}
=
\begin{bmatrix}
40 \\
30 \\
-20
\end{bmatrix}
\]

\[
\Rightarrow v_1 = 2V; \quad v_2 = 30V; \quad v_3 = 24V
\]

