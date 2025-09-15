# Glossary - Begreppsförklaringar

## Allmänna Termer

### **Alpha (α)**
Överavkastning jämfört med marknaden efter justering för risk. Positivt alpha indikerar att strategin genererar värde.

### **Asset Allocation**
Fördelning av kapital mellan olika tillgångar (aktier, obligationer, råvaror etc.) för att optimera risk-avkastning.

### **Backtesting**
Testning av en handelsstrategi på historisk data för att utvärdera dess potential innan implementering.

### **Beta (β)**
Mått på hur mycket en tillgång rör sig relativt marknaden. Beta > 1 = mer volatil än marknaden.

### **Bias**
Systematisk fördom eller fel i analys. T.ex. "confirmation bias" = tendens att söka information som bekräftar ens åsikter.

### **Black Swan**
Extremt sällsynt händelse med stor påverkan som är svår att förutsäga (t.ex. COVID-19, finanskrisen 2008).

## Bayesiansk Analys

### **Bayesian Inference**
**Definition**: Statistisk metod som uppdaterar sannolikheter när ny information blir tillgänglig.
**Formel**: P(H|E) = P(E|H) × P(H) / P(E)
- P(H|E) = Posterior (sannolikhet efter ny information)
- P(E|H) = Likelihood (sannolikhet för data givet hypotes)
- P(H) = Prior (initial sannolikhet)
- P(E) = Evidence (total sannolikhet för data)

### **E[r] - Expected Return**
**Definition**: Förväntad daglig avkastning baserat på Bayesiansk signalkombination.
**Enhet**: Decimal (0.001 = 0.1% daglig avkastning)
**Beräkning**: Viktad summa av signaler justerat för osäkerhet och regim.

### **Pr(↑) - Probability Positive**
**Definition**: Sannolikhet för positiv kursutveckling inom 21 dagar.
**Enhet**: 0-1 (0.65 = 65% sannolikhet)
**Användning**: Primär faktor för köp/sälj-beslut.

### **Prior Beliefs**
**Definition**: Initiala antaganden om signalernas effektivitet innan marknadsdata.
**Exempel**: Momentum effectiveness = 0.68 (68% sannolikhet att momentum-signal är korrekt)

### **Posterior**
**Definition**: Uppdaterad sannolikhet efter att ha observerat marknadsdata.
**Beta Distribution**: Används för att modellera signaleffektivitet med parametrar α (framgångar) och β (misslyckanden).

## Regim-detektion

### **Market Regime**
**Definition**: Rådande marknadsförhållanden som påverkar hur signaler ska tolkas.

#### **Bull Market (Tjurmarknad)**
- **Kännetecken**: Stigande trend, låg volatilitet, optimism
- **Signaljustering**: Förstärkt momentum, standard trend, reducerad sentiment
- **Allokering**: Upp till 100% investerat kapital

#### **Bear Market (Björnmarknad)**
- **Kännetecken**: Fallande trend, hög volatilitet, pessimism
- **Signaljustering**: Reducerad momentum, förstärkt trend och sentiment
- **Allokering**: Max 60% investerat kapital

#### **Neutral Market**
- **Kännetecken**: Sidledes rörelse, måttlig volatilitet, blandade signaler
- **Signaljustering**: Standardviktning för alla signaler
- **Allokering**: Balanserad approach med högre osäkerhetstolerans

### **HMM - Hidden Markov Model**
**Definition**: Statistisk modell för regime-övergångar där det underliggande tillståndet (regim) inte direkt observeras.
**Komponenter**:
- Tillstånd (Bull/Bear/Neutral)
- Övergångssannolikheter
- Emissionssannolikheter (observationer givet tillstånd)

### **Regime Persistence**
**Definition**: Sannolikhet att stanna i samma regim (förhindrar överdriven växling).
**Standard**: 0.80 (80% sannolikhet att stanna kvar)

## Heavy-tail Risk Modeling

### **Fat Tails (Tjocka svansar)**
**Definition**: Högre sannolikhet för extrema händelser än normalfördelning förutspår.
**Exempel**: Börsfall > 5% händer oftare än normalfördelning anger.

### **Student-t Distribution**
**Definition**: Sannolikhetsfördelning som bättre fångar fat tails än normalfördelning.
**Parameter**: ν (degrees of freedom)
- ν → ∞: Närmar sig normalfördelning
- ν < 10: Tjocka svansar (realistiskt för aktier)
- ν ≈ 3-6: Mycket tjocka svansar (högriskaktier)

### **EVT - Extreme Value Theory**
**Definition**: Statistisk teori som specifikt modellerar extrema händelser i svansar.
**Metod**: Peak Over Threshold (POT) - analyserar händelser över vissa trösklar.

### **GPD - Generalized Pareto Distribution**
**Definition**: Fördelning som används i EVT för att modellera svansexcesser.
**Parametrar**:
- ξ (shape): Svansens tyngd
- β (scale): Spridning av excesser

### **VaR - Value at Risk**
**Definition**: Maximal förväntad förlust med given konfidensgrad under bestämd tidsperiod.
**Formel**: P(Loss ≤ VaR) = α (t.ex. α = 0.95 för 95% VaR)
**Exempel**: VaR₉₅% = -5% betyder 5% sannolikhet för förlust större än 5%

### **CVaR - Conditional Value at Risk**
**Definition**: Förväntad förlust givet att förlusten överstiger VaR.
**Även känt som**: Expected Shortfall
**Tolkning**: Genomsnittlig förlust i värsta α% av fallen

### **Tail Risk Multiplier**
**Definition**: Förhållande mellan heavy-tail VaR och normal VaR.
**Beräkning**: Tail Risk Multiplier = VaRₛₜᵤdₑₙₜ₋ₜ / VaRₙₒᵣₘₐₗ
**Tolkning**: 2.0x = heavy-tail risk är dubbelt så stor som normalfördelning anger

## Monte Carlo Simulation

### **Monte Carlo Method**
**Definition**: Simuleringsmetod som använder slumptal för att lösa komplexa matematiska problem.
**Process**:
1. Dra slumptal från fitted Student-t fördelning
2. Applicera drift (förväntad avkastning)
3. Skala för tidshorisont
4. Beräkna statistik över simuleringar

### **Probability Targets**
- **P(return > 0%)**: Sannolikhet för positiv avkastning
- **P(return > +20%)**: Sannolikhet för > 20% uppgång
- **P(return < -20%)**: Sannolikhet för > 20% nedgång

### **Percentiler**
- **1st percentile**: Värsta 1% av utfall
- **99th percentile**: Bästa 1% av utfall

## Portfolio Management

### **Kelly Criterion**
**Definition**: Formel för optimal position sizing baserat på förväntad avkastning och vinstchans.
**Formel**: f* = (bp - q) / b
- f* = optimal andel av kapital
- b = odds (avkastning vid vinst)
- p = sannolikhet för vinst
- q = sannolikhet för förlust (1-p)

### **Risk Parity**
**Definition**: Portföljstrategi där varje position bidrar lika mycket till total risk.
**Implementering**: Viktning = 1/σᵢ / Σ(1/σⱼ) där σ = volatilitet

### **Position Sizing**
**Formula i systemet**:
```
risk_adjusted_return = E[r] × confidence × regime_stability × tail_risk_penalty
weight = (risk_adjusted_return / Σ(risk_adjusted_returns)) × total_budget
```

### **Regime Diversification**
**Definition**: Krav på att positioner ska spridas över olika marknadsregimer.
**Regel**: Max 85% av portföljen får vara i samma regim

## Tekniska Indikatorer

### **SMA - Simple Moving Average**
**Definition**: Genomsnittspris över specificerad period.
**Formel**: SMA = (P₁ + P₂ + ... + Pₙ) / n
**Standard**: 200-dagars SMA för långsiktig trend

### **Momentum**
**Definition**: Kursutveckling över specificerad period.
**Beräkning**: (Pᵢₒday - P₀) / P₀
**Standard**: 252-dagars (ettårs) momentum

### **Mom_rank - Momentum Ranking**
**Definition**: Relativ position av momentum inom universum (0-1 skala).
**Beräkning**: Percentile rank av momentum över alla aktier

## Risk Metrics

### **Sharpe Ratio**
**Definition**: Riskjusterat avkastningsmått.
**Formel**: Sharpe = (E[r] - rf) / σ
- E[r] = förväntad avkastning
- rf = riskfri ränta
- σ = volatilitet

### **Volatility (σ)**
**Definition**: Standardavvikelse av avkastning, mått på prisrörelsers storlek.
**Annualisering**: σₐₙₙᵤₐₗ = σdₐᵢₗᵧ × √252

### **Drawdown**
**Definition**: Maximal förlust från senaste högsta punkt.
**Formel**: DD = (Pₜᵣₒᵤgₕ - Pₚₑₐₖ) / Pₚₑₐₖ

### **Tail Risk Score**
**Definition**: Systemets interna mått på extremrisk (0-1 skala).
**Komponenter**:
- Momentum volatilitet
- Regimjustering
- Signalösäkerhet
**Kategorier**: 🟢 ≤ 0.4 (låg), 🟡 0.4-0.7 (medel), 🔴 > 0.7 (hög)

## Statistiska Begrepp

### **Skewness (Skevhet)**
**Definition**: Mått på asymmetri i fördelning.
- Positiv skewness: Längre höger svans (få mycket positiva värden)
- Negativ skewness: Längre vänster svans (få mycket negativa värden)

### **Kurtosis**
**Definition**: Mått på svansarnas tjocklek.
- Excess kurtosis > 0: Tjockare svansar än normalfördelning
- Excess kurtosis = 0: Normalfördelning
- Excess kurtosis < 0: Tunnare svansar

### **Confidence Interval**
**Definition**: Intervall som med viss sannolikhet innehåller det sanna värdet.
**Exempel**: 95% konfidensintervall [0.02, 0.08] för E[r]

### **P-value**
**Definition**: Sannolikhet att observera resultatet om nollhypotesen är sann.
**Tolkning**: Lågt p-värde (< 0.05) = statistiskt signifikant resultat

## Affärstermer

### **Basis Points (bps)**
**Definition**: Enhet för räntor och avgifter. 1 bps = 0.01%
**Exempel**: 5 bps transaktionskostnad = 0.05% av handelsvärdet

### **Long Position**
**Definition**: Äga en tillgång i förväntan om prisstegring.

### **Short Position**
**Definition**: Låna och sälja tillgång i förväntan om prisfall.

### **Liquidity**
**Definition**: Hur lätt en tillgång kan köpas/säljas utan att påverka priset.

### **Slippage**
**Definition**: Skillnad mellan förväntad exekveringskurs och verklig kurs.

## Systemspecifika Förkortningar

### **ROI**
**Definition**: "Return on Investment" - Systemets namn, avkastning på investering.

### **DoF**
**Definition**: Degrees of Freedom i Student-t fördelning.

### **EVT**
**Definition**: Extreme Value Theory

### **MC**
**Definition**: Monte Carlo (simulation)

### **HMM**
**Definition**: Hidden Markov Model

### **GPD**
**Definition**: Generalized Pareto Distribution

### **POT**
**Definition**: Peak Over Threshold (EVT-metod)

## Matematiska Notationer

### **E[x]**
Väntevärde (expected value) av variabel x

### **Pr(A)**
Sannolikhet för händelse A

### **σ (sigma)**
Standardavvikelse (volatilitet)

### **μ (mu)**
Medelvärde

### **ν (nu)**
Frihetsgrader i Student-t fördelning

### **α (alpha)**
Konfidensgrad eller överavkastning

### **β (beta)**
Marknadsrisk eller Beta distribution parameter

### **ξ (xi)**
Shape-parameter i GPD

### **∑ (sigma)**
Summa över värden

### **∏ (pi)**
Produkt över värden

### **~ (tilde)**
"Är fördelad som", t.ex. X ~ N(μ,σ²)

## Tidsenheter

### **21d/21-day**
21 handelsdagar ≈ 1 månad

### **63d/63-day**
63 handelsdagar ≈ 3 månader (kvartalsvis)

### **252d/252-day**
252 handelsdagar = 1 handelsår

### **12m/12-month**
12 månader = 1 kalenderår

## Beslutskategorier

### **Buy (Köp)**
Signal att öka position i aktien

### **Sell (Sälj)**
Signal att minska eller avveckla position

### **Hold (Behåll)**
Signal att behålla nuvarande position eller stå utanför

### **Uncertainty**
Osäkerhetsmått (0-1) där högre värden indikerar mer osäkerhet i beslutet

### **Confidence**
Beslutssäkerhet (0-1) där högre värden indikerar större förtroende för beslutet