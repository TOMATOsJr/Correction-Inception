![Source Image](9_6-1_source.png)
For \( t > 0 \):

\( RCL \rightarrow Nc/250 + i_L + (5 \times 10^{-6}) \frac{dNc}{dt} = 0 \quad \text{and} \quad Nc = 0.8 \frac{di_L}{dt} \)

Isolate \( i_L \) in \( RCL \) and plug into \( Nc = 0.8 \frac{di_L}{dt} \):

\[
\frac{d^2 Nc}{dt^2} + 800 \frac{di_L}{dt} + (2.5 \times 10^5) Nc = 0
\]

(Characteristic eq.: \( s^2 + 800s + (2.5 \times 10^5) = 0 \))

Roots = \(-400 \pm 300i\)

\( i_L(t) = e^{-400t} [A_1 \cos 300t + A_2 \sin 300t] \)

For \( t = 0^- \):

\( i_L(0^-) = -6/500 = i_L(0^+) \)

\( Nc(0^-) = 250 \cdot (-6/500) + 6 = Nc(0^+) = 3 \)

\( i_L(0) = 3 = A_1 \)

\[
\frac{dNc(0^+)}{dt} = (-2 \times 10^3) i_L(0^+) - 300 Nc(0^+) = 0
\]

\[
\frac{dNc(0^+)}{dt} = 0 = -400 A_1 + 300 A_2 \Rightarrow 300 A_2 = 1200 \quad \Rightarrow A_2 = 4
\]

\[
Nc(t) = e^{-400t} [3 \cos 300t + 4 \sin 300t] \quad \textcolor{red}{[V]}
\]

