```latex
\section{Experimental Results}

In this section, we present the evaluation results of our proposed \textbf{ExTra} algorithm and the baseline \textbf{GRPO} on widely recognized mathematical reasoning benchmarks: MATH-500, AMC23, AIME24, and AIME25. 

To comprehensively assess the mathematical reasoning capabilities and the diversity of exploration, we evaluate the models using two metrics:
\begin{itemize}
    \item $\textbf{pass}@1$: the average success rate of single sampled responses (mean score).
    \item $\textbf{pass}@16$: the success rate when $16$ independent responses are sampled per problem, considering it a success if at least one response is correct (best score).
\end{itemize}

\begin{table}[htbp]
\centering
\caption{Performance Comparison on Mathematical Reasoning Benchmarks. We report the $\text{pass}@1$ (mean score) and $\text{pass}@16$ (best score) across MATH-500, AMC23, AIME24, and AIME25. The metrics reflect the capabilities of the models after 300 steps of RL fine-tuning.}
\label{tab:main_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|cc|cc|cc|cc}
\toprule
\multirow{2}{*}{\textbf{Model \& Algorithm}} & \multicolumn{2}{c|}{\textbf{MATH-500}} & \multicolumn{2}{c|}{\textbf{AMC23}} & \multicolumn{2}{c|}{\textbf{AIME24}} & \multicolumn{2}{c}{\textbf{AIME25}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
& pass@1 & pass@16 & pass@1 & pass@16 & pass@1 & pass@16 & pass@1 & pass@16 \\
\midrule
\textbf{Qwen2.5-1.5B} & & & & & & & & \\
\quad ExTra (Alpha=0.05) & 53.2\% & 82.8\% & 29.8\% & 65.0\% & 2.9\% & 16.7\% & 0.8\% & 10.0\% \\
\midrule
\textbf{Nemotron-1.5B} & & & & & & & & \\
\quad GRPO (Baseline) & 75.6\% & 93.4\% & 65.3\% & 95.0\% & 22.5\% & 53.3\% & 22.5\% & 63.3\% \\
\bottomrule
\end{tabular}
}
\end{table}

As shown in Table~\ref{tab:main_results}, the baseline GRPO applied to Nemotron-1.5B achieves strong performance, reaching $75.6\%$ $\text{pass}@1$ on MATH-500 and demonstrating substantial capabilities on the harder AIME datasets ($22.5\%$ $\text{pass}@1$). The ExTra algorithm applied to Qwen2.5-1.5B reaches $53.2\%$ $\text{pass}@1$ on MATH-500 and $82.8\%$ $\text{pass}@16$. The significant gap between $\text{pass}@1$ and $\text{pass}@16$ for ExTra (e.g., jumping from $2.9\%$ to $16.7\%$ on AIME24) highlights the robust exploratory diversity induced by our novelty-driven intrinsic rewards. 
```