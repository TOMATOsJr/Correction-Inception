![Source Image](9_7-2_source.png)
 

a) \( 24000 \textcolor{red}{=} \frac{d^2 v}{dt^2} + 70 \frac{dv}{dt} + 12000 \textcolor{red}{v} \)

\( 12000(0.2t) = \frac{d^2 v}{dt^2} + 70 \frac{dv}{dt} + 12000 \textcolor{red}{v} \)

\( 2400t = 70A + 12000At + 12000B \)

\[
\begin{cases}
2400 = 12000A & \Rightarrow A = 0.2 \\
70A + 12000B = 0 & \Rightarrow 70(0.2) + 12000B = 0 & \Rightarrow B = -\frac{7}{6000}
\end{cases}
\]

\[
V_f(t) = \frac{1}{5}t \cdot \frac{1}{6000} \textcolor{red}{V}
\]

b) \( i(t) = At + B \)

\( 12000(0.2t) = \frac{di}{dt} + 70 \frac{dV}{dt} + 12000 \textcolor{red}{i} \)

\( 2400t = 70A + 12000At + 12000B \)

\[
\begin{cases}
2400 = 12000A & \Rightarrow A = 0.2 \\
70A + 12000B = 0 & \Rightarrow 70(0.2) + 12000B = 0 & \Rightarrow B = -\frac{7}{6000}
\end{cases}
\]

\[
V_f(t) = \frac{1}{5}t \cdot \frac{1}{6000} \textcolor{red}{V}
\]

c) \( V_f(t) = Ae^{-bt}, \quad \frac{dV_f}{dt} = -ABe^{-bt} \frac{d^2V_f}{dt^2} = +AB^2e^{-2bt} \)

\( 12000(1 - e^{-bt}) = +AB^2e^{-2bt} - 70ABe^{-bt} + 12000Ae^{-bt} \)

\( 12000e^{-bt} = e^{-bt}(AB^2 - 70AAB + 12000A) = 12000 \)

\( B = 30, \quad 70A - 2100A + 12000A = 12000 \)

\(\Rightarrow A = \frac{10}{9}\)

\[
V_f(t) = \frac{10}{9}e^{-30t} \textcolor{red}{V}
\]

