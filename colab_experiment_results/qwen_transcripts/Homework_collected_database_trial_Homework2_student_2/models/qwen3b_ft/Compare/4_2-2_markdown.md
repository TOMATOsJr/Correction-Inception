![Source Image](4_2-2_source.png)
 Node Voltage Analysis

Node 1 "out": 
\[
i_1 + i_2 + i_3 = 0
\]
\[
1 + \frac{v_1 - v_2}{20} + \frac{v_1}{5} = 0 \quad \rightarrow \quad 20 + v_1 - v_2 + 4v_1 = 0
\]

Node 2 "out": 
\[
\frac{v_2 - v_1}{20} - 2 + \frac{v_2 - v_3}{10} = 0 \quad \rightarrow \quad v_2 - v_1 - 40 + 2v_2 - 2v_3 = 0
\]

Node 3 "out": 
\[
-1 + \frac{v_3 - v_2}{10} + \frac{v_3}{5} = 0 \quad \rightarrow \quad -30 + 3v_3 - 3v_2 + 2v_3 = 0
\]

Equations:
\[
5v_1 - v_2 = -20
\]
\[
-v_1 + 3v_2 - 2v_3 = 40
\]
\[
-3v_2 + 5v_3 = 30
\]

\[
\Rightarrow \begin{bmatrix}
5 & -1 & 0 \\
-1 & 3 & -2 \\
0 & -3 & 5
\end{bmatrix} \begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix} = \begin{bmatrix}
-20 \\
40 \\
30
\end{bmatrix}
\]

\[
\rightarrow \begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix} = \begin{bmatrix}
2 \\
30 \\
24
\end{bmatrix}
\]

