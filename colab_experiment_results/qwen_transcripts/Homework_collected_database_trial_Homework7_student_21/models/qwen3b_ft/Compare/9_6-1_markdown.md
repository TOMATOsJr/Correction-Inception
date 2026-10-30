![Source Image](9_6-1_source.png)
 

\[
t > 0 \quad \textcolor{red}{V = 1 \frac{dI}{dt}} \quad i = C \frac{dV}{dt}
\]

\[
\begin{array}{|c|c|c|}
\hline
\textcolor{red}{i} & \textcolor{red}{V} & i = C \frac{dV}{dt} \\
\hline
\textcolor{red}{i} & \textcolor{red}{0.8 \text{ A}} & \textcolor{red}{i} = C \frac{dV}{dt} \\
\hline
\textcolor{red}{250 \Omega} & \textcolor{red}{5 \times 10^5 \text{ F}} & \textcolor{red}{\frac{V}{R} + \frac{V}{E_1} + CSV = 0} \\
\hline
\textcolor{red}{\frac{V}{R}} & \textcolor{red}{\frac{V}{E_1} + CSV = 0} & \textcolor{red}{\frac{V}{P} SV + V + LGS^2V = 0} \\
\hline
\textcolor{red}{(S^2 + \frac{1}{RC}CS + \frac{1}{LC})V = 0} & \textcolor{red}{no forced} & \textcolor{red}{S^2 + 800S + 250000 = 0} \quad \textcolor{red}{\text{resonance}} \\
\hline
\textcolor{red}{S = 400} & \textcolor{red}{W_0^2 = 250,000} \quad \textcolor{red}{\text{underdamped}} & \textcolor{red}{W_0 = \sqrt{250,000 - 400^2} = 300} \\
\hline
\textcolor{red}{V(t)} = e^{-\frac{t}{400}} (B_1 \cos 300t + B_2 \sin 300t) & & \textcolor{red}{\text{O} = 300B_2 - 1200} \\
\hline
\end{array}
\]

\[
t < 0 \quad \textcolor{red}{6V} = 0.012 \text{ A} \rightarrow 250 \text{ n} \cdot (0.012 \text{ A}) = 3 \text{ V} \\
\textcolor{red}{3V = e^{-\frac{t}{400}} (B_1 \cos 300t + B_2 \sin 300t)} \\
\textcolor{red}{\textcolor{red}{250 \Omega}} \quad \textcolor{red}{\textcolor{red}{250 \Omega}} \quad \textcolor{red}{\textcolor{red}{6V}} \quad \textcolor{red}{C \frac{dV}{dt} = e^{-\frac{t}{400}} (-900 \sin 300t + 300B_2 \cos 300t)} \\
\textcolor{red}{\textcolor{red}{= 400e^{-\frac{t}{400}} (3 \cos 300t + B_2 \sin 300t)}} \\
\textcolor{red}{\textcolor{red}{\text{O} = 300B_2 - 1200}} \\
\textcolor{red}{B_2 = 4} \\
\textcolor{red}{V(t) = e^{-\frac{t}{400}} (3 \cos 300t + 4 \sin 300t)} \\
\end{array}
\]

