![Source Image](8_3-1_source.png)
 

\[
t > 0 \quad V_1 = V(6)
\]

\begin{tabular}{c c}
\begin{tikzpicture}[scale=0.8]
    \draw[thick, ->] (-2, 0) -- (2, 0) node[right] {$6\Omega$};
    \draw[thick, ->] (0, 0) -- (0, 2) node[above] {$12V$};
    \draw[thick, ->] (0, 0) -- (0, -2) node[below] {$-4V$};
    \draw[thick, ->] (0, 0) -- (2, 2) node[above] {$V_1$};
    \draw[thick, ->] (0, 0) -- (-2, 2) node[left] {$V(6)$};
\end{tikzpicture}
\end{tabular}

\textcolor{red}{k(t)} @ $V_1$ "out"

\[
\frac{V - 12}{6} + \frac{V}{6} + k \frac{d}{dt} V_1 = 0
\]

\[
\frac{2V_1}{6} + k \frac{d}{dt} V_1 = 2 \quad \Rightarrow \quad k = \frac{1}{4} F
\]

\[
\frac{d}{dt} V_1 + \frac{4}{3} V_1 = 0
\]

\[
T = RC
\]

\[
R = \frac{V}{RC}
\]

\[
kT = V
\]

