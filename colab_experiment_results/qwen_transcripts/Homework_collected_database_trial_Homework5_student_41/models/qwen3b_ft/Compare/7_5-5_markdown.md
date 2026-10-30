![Source Image](7_5-5_source.png)
 

\[
i_L(t) = \frac{1}{L} \int_0^t v_L(\tau) d\tau + i_L(1\text{ s})
\]

\begin{itemize}
    \item \textcolor{red}{For \(0 \leq t \leq 1\) ms}
    \[
    i(t) = \frac{1}{5 \times 10^{-3}} \int_0^{1 \times 10^{-3}} 4 \times 10^{-3} d\tau - 2 \times 10^{-6}
    \]
    \[
    = \frac{1}{5} (4t) - 2 \times 10^{-6}
    \]
    \[
    = 0.8t - 2 \times 10^{-6} A
    \]
    \[
    i(1 \times 10^{-6}) = -1.2 \, mA
    \]
    
    \item \textcolor{red}{For \(1 < t \leq 3\) ms}
    \[
    i(t) = \frac{1}{5 \times 10^{-3}} \int_{1 \times 10^{-3}}^{3 \times 10^{-3}} -1 \times 10^{-3} d\tau - 1.2 \times 10^{-6}
    \]
    \[
    = -0.2t - 1.2 \times 10^{-6} A
    \]
    \[
    i(3 \times 10^{-6}) = -0.2(3 \times 10^{-6}) - 1.2 \times 10^{-6} A = -1.8 \, mA
    \]
\end{itemize}

\[
i(t) = 
\begin{cases}
0.8t - 2 \times 10^{-6} A, & 0 \leq t \leq 1 \text{ ms} \\
-0.2t - 1.2 \times 10^{-6} A, & 1 < t \leq 3 \text{ ms} \\
-0.0018 A, & t > 3 \text{ ms}
\end{cases}
\]

