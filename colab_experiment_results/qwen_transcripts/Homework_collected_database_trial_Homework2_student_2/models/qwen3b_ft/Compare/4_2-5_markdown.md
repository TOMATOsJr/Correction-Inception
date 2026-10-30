![Source Image](4_2-5_source.png)
 Node Voltage Analysis!

\[
\textcolor{red}{\text{Node 1 "out":}} \quad \frac{v_1 - v_5}{250} + \frac{v_1 - v_3}{125} + \frac{v_1}{500} = 0 \quad \rightarrow \quad 2v_1 - 2v_3 - 4v_1 - 4v_2 + v_1 = 0
\]

\[
\textcolor{red}{\text{Node 2 "out":}} \quad \frac{v_1 - v_4}{125} - 1 + \frac{v_2 - v_3}{250} = 0 \quad \rightarrow \quad 2v_2 - 2v_1 - 250 + r v_2 - v_3 = 0
\]

\[
\textcolor{red}{\text{Node 3 "out":}} \quad \frac{v_3 - v_4}{250} + \frac{v_3 - v_2}{250} + \frac{v_3}{500} = 0 \quad \rightarrow \quad 2v_3 - 2v_1 + 2v_3 - 2v_2 + v_3 = 0
\]

Matrix

\[
\begin{bmatrix}
7 & -4 & -2 \\
-2 & 3 & -1 \\
-2 & -2 & 5
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix}
=
\begin{bmatrix}
0 \\
250 \\
0
\end{bmatrix}
\quad \rightarrow \quad 
\begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix}
=
\begin{bmatrix}
0 \\
250 \\
0
\end{bmatrix}
\quad \rightarrow \quad 
\begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix}
=
\begin{bmatrix}
260.8096 \\
336.9505 \\
2391.304
\end{bmatrix}
\]

\[
V = V_1 - V_3 \quad \rightarrow \quad V = 21.731 \, \text{mV}
\]

