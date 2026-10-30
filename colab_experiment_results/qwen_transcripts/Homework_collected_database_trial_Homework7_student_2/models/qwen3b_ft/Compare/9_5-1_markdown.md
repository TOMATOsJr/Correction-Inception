![Source Image](9_5-1_source.png)
 

\begin{align*}
t < 0 & \quad i_L = 30 \, \text{mA} \\
& \quad v_C = 0 \\
& \quad i_L(0) = 0 \\
& \quad v_C(0) = 0.3 \, (\textcolor{red}{100}) = 3 \, \text{V} \\
\end{align*}

\textcolor{red}{Initial condition!} \\
\( v_C(0) = 3 = C_1 \)

\[
i_L = C \cdot v'(0)
\]

\[
v'(t) = e^{-2000t} \left[ -6000C_1 \sin(6000t) + 6000C_2 \cos(6000t) \right]
\]

\[
-2000e^{-2000t} \left[ C_1 \cos(6000t) + C_2 \sin(6000t) \right]
\]

\[
v'(0) = 1 \left[ 6000C_2 - 2000(C_1) \right]
\]

\[
v'(0) = 6000C_2 - 6000 = 0
\]

\[
C_2 = 1
\]

\[
v(t) = e^{-2000t} \left[ 3 \cos(6000t) + \sin(6000t) \right]
\]

