![Source Image](4_2-5_source.png)
 KCL @ node I

\[
\frac{V_1}{500} + \frac{V_1 - V_2}{125} + \frac{V_1 - V_3}{250} = 0
\]

KCL @ node Z

\[
\frac{V_2}{250} + \frac{V_2 - V_3}{250} = \frac{V_3}{500}
\]

\[
\frac{V_2}{250} + \frac{V_2 - V_3}{250} + \frac{V_1 - V_3}{250} = 0
\]

\[
\frac{V_1}{500} + \frac{V_1}{125} - \frac{V_2}{125} + \frac{V_3}{250} = 0
\]

\[
\frac{V_1}{500} + \frac{V_1}{125} - \frac{V_2}{125} - \frac{V_3}{250} = 0
\]

\[
\frac{2V_1}{500} - \frac{V_2}{125} - \frac{V_3}{250} = 0
\]

\[
\begin{pmatrix}
\frac{1}{1500} & -\frac{1}{125} & -\frac{1}{125} \\
\frac{1}{125} & -\frac{1}{750} & \frac{1}{250} \\
\frac{1}{125} & \frac{1}{750} & -\frac{1}{100}
\end{pmatrix}
N
\begin{pmatrix}
10 \\
0 \\
0
\end{pmatrix}
=
\begin{pmatrix}
0.26086 \\
0.13494 \\
0.2391
\end{pmatrix}
\]

\[
V = V_1 - V_3 = 0.26086V - 0.2391V = 21.7mV
\]

