![Source Image](9_6-1_source.png)
 

\[
R_1 = 250 \Omega, R_2 = 250 \Omega, L = 0.1 \textcolor{red}{H}, C = 5 \times 10^{-6} \textcolor{red}{F}
\]

\[
V_1 = 6 \textcolor{red}{V}
\]

\[
\begin{cases}
\frac{dV_c}{dt} + i_1 + \frac{1}{R_1} V_c = 0 \\
L \cdot \frac{di_1}{dt} = V_c
\end{cases}
\]

\[
\Rightarrow i_1 = (-C \cdot s \cdot V_c - \frac{1}{R_1} V_c)
\]

\[
V_c = L \cdot s \cdot i_1
\]

\[
V_c = L \cdot s \cdot \left( -C \cdot s \cdot V_c - \frac{1}{R_1} V_c \right)
\]

\[
0 = V_c \cdot (L \cdot s^2 - \frac{1}{R_1} s^2 + 1) \Rightarrow s^2 \cdot \frac{L}{R_1} (s^2 - \frac{1}{LC}) = 0
\]

\[
\Rightarrow 0 = s^2 \cdot 800 \textcolor{red}{s} + 250000 \Rightarrow s = \frac{-800 \pm \sqrt{1000000}}{250000} = -400 \pm 300j
\]

\[
V_c(t) = e^{-400t} \left( A_1 \cos(300t) + A_2 \sin(300t) \right)
\]

\[
\frac{dV_c(t)}{dt} = -400A_1 e^{-400t} \cos(300t) + A_1 e^{-400t} (-\sin(300t)) = -400A_1 e^{-400t} \sin(300t) + 300A_2 e^{-400t} \cos(300t)
\]

\[
V_c(0) = A_1 \Rightarrow \frac{dV_c(0)}{dt} = -400A_1 + 300A_2
\]

Before \( t = 0 \): (capacitor = open circuit, inductor = short circuit)

\[
R = R_1 + R_2 = 500 \Omega
\]

\[
-i(0) = \frac{6 \textcolor{red}{V}}{500 \Omega} \Rightarrow i(0)^2 = -0.12 \textcolor{red}{A}
\]

\[
V_c(0) = -i(0)(0) \cdot 250 \Omega = 3 \textcolor{red}{V} \Rightarrow A_1 = 3
\]

\[
\frac{dV_c(t)}{dt} = (-400A_1 + 300A_2) \cdot (s \cdot 10^{-3} \textcolor{red}{F})
\]

\[
0 = -400(3) + 300A_2 \Rightarrow A_2 = 4
\]

\[
\therefore V_c(t) = e^{-400t} \left( 3 \cos(300t) + 4 \sin(300t) \right) \textcolor{red}{V}
\]

