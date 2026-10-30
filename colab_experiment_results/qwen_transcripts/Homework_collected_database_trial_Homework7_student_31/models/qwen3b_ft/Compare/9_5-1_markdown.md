![Source Image](9_5-1_source.png)
For \( x > 0 \):

KVL \(\Rightarrow\) \( 100 i_c + 0.25 \frac{d i_c}{dx} + v_c = 0 \) and \( i_c = 10^{-5} \frac{dv_c}{dx} \)

\(\Rightarrow\) \( \frac{d^2 v_c}{dx^2} + 4000 \frac{d i_c}{dx} + (4 \times 10^6) i_c = 0 \)

Charateristic: \( s^2 + 4000s + (4 \times 10^6) = 0 \)

Roots \(\Rightarrow\) \( s_{1,2} = -2000 \)

\( i_c(x) = A_1 e^{-2000x} + A_2 e^{-2000x} \)

For \( x = 0^- \):

\( i_L = i_c(0^-) = 0 = i_c(0^+) \Rightarrow \frac{d v_c(0^+)}{dx} = 0 \)

\( v_c(0^-) = 3 = v_c(0^+) = A_1 \)

\( \frac{d v_c(0^+)}{dx} = 0 = -2000A_1 + A_2 \Rightarrow A_2 = 6000 \)

\(\therefore v_c(x) = (3 + 6000x)e^{-2000x} \quad \textcolor{red}{V}\)

