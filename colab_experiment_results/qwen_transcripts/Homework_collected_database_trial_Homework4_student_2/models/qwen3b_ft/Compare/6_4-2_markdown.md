![Source Image](6_4-2_source.png)
 

\textcolor{red}{MCA}
\[
\begin{bmatrix}
12 - G_{ii} - G_{i1} - G_{i2} & 0 \\
12 - G_{ii} - G_{i1} - G_{i2} & 1 \\
-12i_1 - G_{i1} - G_{i2} & -12 \\
6(i_1 - i_2) & 0 \\
G_{ii} - 12i_2 & 0 \\
\end{bmatrix}
\]

\[
\begin{bmatrix}
6(i_1 - i_2) - 6i_2 = 0 \\
G_{ii} - 12i_2 = 0 \\
-12i_1 - G_{i1} - G_{i2} = -12i_2 \\
r_2(G_{ii} - 12i_2 = 0) \\
0 - 18i_2 = 12 \\
\end{bmatrix}
\]

\[
i_2 = \textcolor{red}{0.667 mA}
\]

\textcolor{red}{NVD}

\textcolor{red}{KCL node A}
\[
\frac{V_A - 12}{6} + \frac{V_A}{0} = 0
\]
\[
\frac{V_A - 12}{6} + \frac{-V_A}{0} = 0
\]
\[
V_A - 12 + V_A + V_A = 0
\]
\[
3V_A = 12 \rightarrow V_A = 4V
\]

\textcolor{red}{KCL e output node}
\[
\frac{V_e}{0} + i_e + \frac{V_o}{0} = 0
\]
\[
i_e = -\frac{2V_o}{0}
\]
\[
V_o = -\frac{1}{2}(-4)
\]
\[
V_o = 1.33 \textcolor{red}{mA}
\]

