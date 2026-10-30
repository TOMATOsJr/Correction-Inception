![Source Image](9_4-1_source.png)
 

\[
\textcolor{red}{KVL}
\]

\[
-i_L(t) + \frac{v_C(t)}{80} + i_c(t) = 0
\]

\[
-i_L(t) + \frac{v_C(t)}{80} + C \frac{dv_C}{dt} = 0
\]

\[
i_L(t) = \frac{v_C(t)}{80} + C \frac{dv_C}{dt}
\]

\[
i_L(t) = i
\]

\[
v_C(t) = V
\]

\[
\frac{d}{dt} V = S
\]

\[
i = \frac{1}{80} V = CS V
\]

\[
\textcolor{red}{Initial Condition}
\]

\[
v(0) = A + B
\]

\[
-3000 = -400A - 100B
\]

\[
\begin{bmatrix}
1 & 1 \\
-400 & -3000
\end{bmatrix}
\Rightarrow
\begin{bmatrix}
A \\
B
\end{bmatrix}
=
\begin{bmatrix}
8 \\
2
\end{bmatrix}
\]

\[
v(t) = 8 e^{-400t} - 2 e^{-100t}
\]

\[
\textcolor{red}{Characteristic Eqn}
\]

\[
S^2 + 500S + 40000 = 0
\]

\[
S = \frac{-500 \pm \sqrt{(500)^2 - 4 \cdot 40000}}{2}
\]

\[
S = \frac{-500 \pm 300}{2}
\]

\[
S_1 = -400, \quad S_2 = -100
\]

