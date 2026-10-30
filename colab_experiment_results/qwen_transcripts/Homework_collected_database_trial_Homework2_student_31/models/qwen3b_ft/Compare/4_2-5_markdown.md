![Source Image](4_2-5_source.png)
 

\[
v = v_1 - v_3
\]

KCL @ Node \( v_1 \):

\[
\frac{(v_1 - v_2)}{250} + \frac{(v_1 - v_2)}{125} + \frac{v_1}{500} = 0
\]

KCL @ Node \( v_2 \):

\[
\frac{(v_1 - v_2)}{125} + 0.001 = \frac{(v_2 - v_3)}{250}
\]

KCL @ Node \( v_3 \):

\[
\frac{(v_1 - v_3)}{250} + \frac{(v_2 - v_3)}{250} = v_3 / 500
\]

\[
7v_1 - 4v_2 - 2v_3 = 0
\]

\[
-2v_1 + 3v_2 - v_3 = 0.25
\]

\[
2v_1 + 2v_2 - 5v_3 = 0
\]

\[
\begin{bmatrix}
7 & -4 & -2 \\
-2 & 3 & -1 \\
2 & 2 & -5
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0.25 \\
0
\end{bmatrix}
\]

\[
\Rightarrow
\begin{bmatrix}
v_1 \\
v_2 \\
v_3
\end{bmatrix}
=
\begin{bmatrix}
6.23 \\
7.75/23 \\
5.5/23
\end{bmatrix}
\]

\[
v = v_1 - v_3 = \frac{6}{23} - \frac{5.5}{23} = \frac{0.5}{23} = [21.7 \textcolor{red}{- mV}]
\]

