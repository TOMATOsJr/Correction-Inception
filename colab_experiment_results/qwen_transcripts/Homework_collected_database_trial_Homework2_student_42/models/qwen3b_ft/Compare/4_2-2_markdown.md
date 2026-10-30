![Source Image](4_2-2_source.png)
 

KCL @ Node I: \textcolor{red}{"out"} 

\[
1 + \frac{v_1 - v_2}{20} + \frac{v_1}{5} = 0 
\]

\[
1 + \frac{v_1}{20} - \frac{v_1}{20} + \frac{v_1}{3} = 0 
\]

\[
20v_1 - v_2 + 4v_1 = 0 
\]

KCL @ Node 2: \textcolor{red}{"in"} 

\[
2 + \frac{v_1 - v_2}{20} = \frac{v_2 - v_3}{10} 
\]

\[
2 + \frac{v_1}{20} - \frac{v_1}{20} = \frac{v_1}{10} - \frac{v_3}{10} 
\]

\[
40v_1 - v_1 - v_2 = 2v_2 - 2v_3 
\]

KCL @ Node 3: \textcolor{red}{"in"} 

\[
1 + \frac{v_1}{10} - \frac{v_3}{10} = \frac{v_2}{15} 
\]

\[
1 + \frac{v_1}{10} - \frac{v_3}{10} = \frac{v_2}{15} 
\]

\[
30v_2 - 3v_3 - 2v_3 = -30 
\]

\[
v_1 - 3v_2 + 2v_3 = -10 
\]

\[
3v_2 - 5v_3 = -30 
\]

\[
\begin{pmatrix}
5 & -1 & 0 \\
1 & -3 & 2 \\
0 & 3 & -5 
\end{pmatrix}
\sim
\begin{pmatrix}
16 & 0 & 2 \\
0 & 10 & 30 \\
0 & 1 & 24 
\end{pmatrix}
\]

