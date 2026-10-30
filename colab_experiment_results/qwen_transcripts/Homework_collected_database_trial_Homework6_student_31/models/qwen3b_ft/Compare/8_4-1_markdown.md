![Source Image](8_4-1_source.png)
 When \( t > 1.5 \) s and \( t < 0 \), the circuit can be reduced to:

\[
\textcolor{red}{\begin{array}{c}
\textcolor{red}{10V} \\
\textcolor{red}{8\Omega} \\
\textcolor{red}{0.05F} \\
\textcolor{red}{v(t)}
\end{array}} \Rightarrow \textcolor{red}{\begin{array}{c}
\textcolor{red}{10V} \\
\textcolor{red}{8\Omega} \\
\textcolor{red}{0.05F} \\
\textcolor{red}{v(t)}
\end{array}}
\]

When \( 0 \leq t < 1.5 \):

\[
\textcolor{red}{\begin{array}{c}
\textcolor{red}{1.732} \\
\textcolor{red}{8\Omega} \\
\textcolor{red}{0.05F} \\
\textcolor{red}{0.4\Omega} \\
\textcolor{red}{0.05F} \\
\textcolor{red}{v(t)}
\end{array}} \Rightarrow \textcolor{red}{\begin{array}{c}
\textcolor{red}{1.732} \\
\textcolor{red}{8\Omega} \\
\textcolor{red}{0.05F} \\
\textcolor{red}{0.4\Omega} \\
\textcolor{red}{0.05F} \\
\textcolor{red}{v(t)}
\end{array}} \Rightarrow \textcolor{red}{\begin{array}{c}
\textcolor{red}{5V} \\
\textcolor{red}{4\Omega} \\
\textcolor{red}{0.05F} \\
\textcolor{red}{v(t)}
\end{array}}
\]

Since circuit will reach steady state we know \( v(0) = 10 \, \text{V} \).

When \( 0 \leq t < 1.5 \), \( R_x = 4 \, \Omega \) and \( V_{oc} = 5 \, \text{V} \).

\[
I = R_x \cdot C = 4 \times 0.05 = 0.2 \, \text{A}
\]

\[
v(t) = V_{oc} + (v(0) - V_{oc}) e^{-5t} = \left[ 5 + 5 e^{-5t} \right] \, \text{V} \quad \text{for } 0 \leq t < 1.5 \, \text{s}
\]

\[
v(1.5) \approx 5 \, \text{V}
\]

When \( t > 1.5 \), \( V_{oc} = 10 \, \text{V} \) and \( R_x = 8 \, \Omega \).

\[
I = R_x \cdot C = 8 \times 0.05 = 0.4 \, \text{A}
\]

\[
v(t) = 10 + (5 - 10) e^{-7(t - 1.5)} / I = \left[ 10 - 5 e^{-2.5(t - 1.5)} \right] \, \text{V} \quad \text{for } t > 1.5 \, \text{s}
\]

\[
v(t) = 
\begin{cases}
5 + 5 e^{-5t} \, \text{V}, & 0 \leq t < 1.5 \, \text{s} \\
10 - 5 e^{-2.5(t - 1.5)} \, \text{V}, & t > 1.5 \, \text{s}
\end{cases}
\]

