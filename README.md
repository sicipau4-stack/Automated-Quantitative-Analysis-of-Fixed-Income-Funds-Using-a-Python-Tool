# Automated-Quantitative-Analysis-of-Fixed-Income-Funds-Using-a-Python-Tool

This project is a Python-based quantitative analysis tool designed to automate the analysis and comparison of fixed-income investment funds. The tool processes historical fund data, including yield to maturity (TIR), duration, prices, and returns, transforming raw financial data into structured datasets ready for quantitative analysis.

The program allows users to select the funds they want to analyse using their ISIN codes and automatically processes the corresponding historical information. It performs data cleaning, standardisation, resampling, interpolation, and validation before carrying out the different analyses. It also generates historical datasets for each selected fund, making the information easier to work with and compare.

The tool provides a wide range of quantitative metrics and visual analyses, including the historical evolution of yield and duration, price and monthly return evolution, average yield and duration, distributions through boxplots, correlations between fund returns, rolling volatility of yield and returns, and the distribution of monthly changes in yield. It also incorporates benchmark indices to provide a reference point for evaluating fund behaviour.

In addition, the project includes more advanced time-series analysis. It automatically applies ARIMA-based modelling to fund returns and calculates Ft adapted volatility using a 36-month window and an estimated parameter based on standardised shocks. This allows the analysis to capture changes in risk and volatility over time rather than relying only on traditional volatility measures.

One of the main features of the project is the **automatic generation of a complete analytical report**. Once the analysis is executed, the program automatically creates the different charts, calculates the relevant metrics and statistical results, and integrates them into a structured report. The report can be generated in Markdown, HTML, and PDF formats, providing a complete overview of the selected funds without requiring the user to manually perform each analysis.

The tool is particularly useful for users who do not have extensive knowledge of investment funds or quantitative finance. Instead of manually collecting data, calculating financial metrics, creating graphs, comparing funds, and interpreting multiple datasets, the user can run the tool and obtain a comprehensive analysis automatically. This significantly reduces the time required to analyse fixed-income funds while making the process more accessible, systematic, and reproducible.

Overall, the project combines **Python, financial data processing, statistical analysis, time-series modelling, volatility analysis, data visualisation, and automated reporting** into a single workflow designed to simplify and accelerate the quantitative analysis of fixed-income funds.
