![Source Image](8_7-5_source.png)
 After \( t = 0 \),

\[
i(t) = \frac{\textcolor{red}{\text{d}v_c(t)}}{15000} = \frac{15 e^{-2t}}{15000} = 0.1 e^{-2t} \, \text{mA}
\]

\[
i(t) = \left( \frac{\textcolor{d}{{\text{d}v_c(t)}}}{R} + v_c(t) \right)
\]

\(\Rightarrow\) 

\[
(i \times 10^3) e^{-2t} = (0.25 \times 10^3) \frac{\textcolor{d}{\text{d}v_c(t)}}{\textcolor{d}{\text{d}t}} + (10^{-3}) v_c(t)
\]

\(\Rightarrow\) 

\[
\frac{\textcolor{d}{\text{d}v_c(t)}}{\textcolor{d}{\text{d}t}} + 4000 v_c(t) = 4000 e^{-2t}
\]

\(\Rightarrow\) 

\[
v_f(k) = B e^{-2t}
\]

\(\Rightarrow\) 

\[
\frac{\textcolor{d}{\text{d}(Be^{-2t})}}{\textcolor{d}{\text{d}t}} + 4000(B e^{-2t}) = 4000 e^{-2t}
\]

\(\Rightarrow\) 

\[
-2Be^{-2t} + 4000Be^{-2t} = 4000 e^{-2t}
\]

\(\Rightarrow\) 

\[
B = \frac{4000}{3998} \approx 1
\]

\(\Rightarrow\) 

\[
v_c(t) = Ae^{-4000t} + e^{-2t}
\]

\(\Rightarrow\) 

\[
v_c(0) = A + 1 = -10 \quad \Rightarrow \quad A = -11
\]

\(\Rightarrow\) 

\[
v_c(t) = e^{-2t} - 11 e^{-4000t}
\]

\(\Rightarrow\) 

\[
v_o(t) = -v_c(t) = 11 e^{-4000t} - e^{-2t} \, \text{V}, \quad \lambda \geq 0
\]

