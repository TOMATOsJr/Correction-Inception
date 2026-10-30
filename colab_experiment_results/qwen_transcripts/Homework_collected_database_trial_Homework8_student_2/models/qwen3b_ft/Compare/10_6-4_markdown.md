![Source Image](10_6-4_source.png)
Mesh current analysis

\[
-(10 + j15)I_1 - (25 - j50)(I_1 - I_2) \cdot 48 e^{j\pi} = 0
\]

\[
-(40 + j15)I_1 - (25 - j50)r, r(25 - j50)I_3 = -48r e^{j\pi}
\]

\[
-(65 - 3j5)I_1 + (25 - j50)I_3 = -4r e^{j\pi}
\]

KVL mesh 2

\[
-48 e^{j\pi}T_1 + j50R_2 - (32 + j10)I_2 = 0
\]

\[
-(32 + 3j4)I_2 = 4r e^{j\pi}
\]

KVL mesh 3

\[
-j40I_3 - j50(I_2 - I_3) + (25 - j50)(I_3 - T_3) = 0
\]

\[
-j40I_3 - j50I_2 + j50I_3 + (25 - j50)I_2 - (25 - j50)T_3 = 0
\]

\[
(25 - j100)I_2 + j10I_3 + (-25 - j50)I_3 = 0
\]

\[
(25 - j100)I_2 + (-25 + j50)I_3 = 0
\]

\[
\begin{bmatrix}
-48 & 0 & 25 - j50 \\
0 & -32 - 3j4 & 0 \\
0 & 25 - j100 & -25 + j50
\end{bmatrix}
\begin{bmatrix}
I_1 \\
I_2 \\
I_3
\end{bmatrix}
=
\begin{bmatrix}
-48 e^{j\pi} \\
48 e^{j\pi} \\
0
\end{bmatrix}
\]

