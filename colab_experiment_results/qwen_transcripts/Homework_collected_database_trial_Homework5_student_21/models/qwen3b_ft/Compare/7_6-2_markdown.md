![Source Image](7_6-2_source.png)
 

\[
p(t) = v(t)i_c(t) = \left( L \frac{di_c(t)}{dt} \right)i_c(t)
\]

\[
v(t) = L \frac{di_c(t)}{dt}
\]

\[
i_c(t) = 
\begin{cases} 
0 & t < 0 \\
40 \cos(2t) & t \geq 0 
\end{cases}
\]

\[
p(t) = v(t)i_c(t)
\]

\[
p(0^-) = 0
\]

\[
p(0^+) = 40 \cos(2t) \cdot 4 \sin(2t) = 160 \cos(2t) \sin(2t)
\]

\[
\text{double angle formula} = 80 \sin(4t)
\]

\[
p(t) = 
\begin{cases} 
0 & t < 0 \\
80 \sin(4t) & t \geq 0 
\end{cases}
\]

\[
w(0^-) = 0
\]

\[
w(0^+) = \int_0^\infty 80 \sin(4t) \, dt = -20 \cos(4t) \Bigg|_0^\infty = -20 [\cos(4t) - \cos(0)] = -20(1 - \cos(4t))
\]

\[
w(t) = 
\begin{cases} 
0 & t < 0 \\
-20(1 - \cos(4t)) & t \geq 0 
\end{cases}
\]

