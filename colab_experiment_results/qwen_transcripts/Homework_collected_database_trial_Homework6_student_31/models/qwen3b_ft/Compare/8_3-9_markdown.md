![Source Image](8_3-9_source.png)
Before switch closes:

\[
\textcolor{red}{\lambda(t) = \frac{1}{R_1}(t)} = \frac{6 - 3e^{-0.35t}}{3} = 2 - e^{-0.35t}
\]

\[
\textcolor{red}{i(0) = 2 - e^{0} = 1 \text{ A}}
\]

KVL:

\[
-24 + R_1 i(0) + R_2 i(0) + 3i(0) = 0
\]

\[
i(0) = 2 - e^{0} = 1 \text{ A} \quad \text{(plug in)}
\]

\[
\Rightarrow -24 + R_1 \cdot 1 + R_2 \cdot 1 + 3 \cdot 1 = 0
\]

\[
\Rightarrow R_1 + R_2 = 21
\]

Long time after switch closes:

\[
\textcolor{red}{\lambda(\infty) = 2 - e^{-\infty} = 2 \text{ A}}
\]

\[
\Rightarrow -24 + 2R_2 + 16 = 0
\]

\[
\Rightarrow 2R_2 = 18 \quad \Rightarrow \quad R_2 = 9 \Omega
\]

\[
\Rightarrow R_1 = 21 - 9 = 12 \Omega
\]

\[
\therefore R_2 = R_1 + 3 = 12 \Omega + 3 = 15 \Omega
\]

\[
\tau = \frac{L}{R_2} = \frac{L}{12}
\]

\[
\therefore L = 34.29 \text{ H}
\]

\[
\textcolor{red}{\text{From } i(t):} \quad -0.35t = -\frac{1}{\tau} t \quad \Rightarrow \quad \tau = 2.86
\]

\[
\therefore L = 34.29 \text{ H}
\]

\textcolor{red}{Final Answer:} 
\[
R_1 = 12 \Omega; R_2 = 9 \Omega; L = 34.29 \text{ H}
\]

