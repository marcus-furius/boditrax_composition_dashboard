# Body Recomposition Dashboard

A comprehensive Dash/Plotly dashboard for tracking body composition changes over time. Designed for fitness enthusiasts pursuing **body recomposition** - the simultaneous process of building muscle while losing fat.

## 🎯 What is Body Recomposition?

Body recomposition challenges the traditional "bulk and cut" approach. Instead of gaining weight (muscle + fat) then losing weight (fat + some muscle), recomposition aims to:
- **Build lean muscle mass** through resistance training and adequate protein
- **Reduce body fat** through caloric management and cardiovascular exercise
- **Maintain or slightly change total body weight** while dramatically improving body composition

This dashboard helps you track whether you are successfully achieving this delicate balance.

---

## 📊 Dashboard Visualizations Guide

### 1️⃣ **KPI Cards - Your Progress at a Glance**

Four primary metrics displayed at the top of the dashboard:

#### 💪 Muscle Mass
- **What it shows**: Total skeletal muscle mass in kilograms
- **How to read**: 
  - Main number: Current muscle mass
  - Δ value: Total change since first scan
  - Week-over-week (WoW): Change from last week
- **Why it matters**: 
  - Muscle is metabolically active tissue - burns calories at rest
  - More muscle = higher basal metabolic rate (BMR)
  - Primary indicator of strength training effectiveness
  - **Target**: Steady increase (0.25-0.5 kg/month is excellent natural progress)

#### 🔥 Fat Mass
- **What it shows**: Total body fat in kilograms
- **How to read**:
  - Main number: Current fat mass
  - Δ value: Total change since first scan (negative is good!)
  - WoW: Weekly change trend
- **Why it matters**:
  - Excess fat increases health risks (cardiovascular disease, diabetes, inflammation)
  - Reducing fat improves insulin sensitivity and hormonal balance
  - Visible indicator of aesthetic progress
  - **Target**: Gradual decrease (0.5-1 kg/month is sustainable)

#### ⚖️ Body Weight
- **What it shows**: Total body weight in kilograms
- **How to read**:
  - During recomp, this may stay relatively stable
  - Small changes are normal and expected
- **Why it matters**:
  - **The least important metric during recomp!**
  - Weight alone does not distinguish muscle from fat
  - Can be misleading (muscle is denser than fat)
  - Use alongside muscle/fat metrics for full picture

#### 🎯 Recomposition Ratio
- **What it shows**: Muscle gained ÷ Fat lost (as a ratio)
- **How to read**:
  - **1.0:1** = Elite (gained 1kg muscle for every 1kg fat lost)
  - **0.7-0.9:1** = Excellent recomposition
  - **0.5-0.7:1** = Good progress
  - **< 0.5:1** = More fat loss than muscle gain (acceptable during cut)
- **Why it matters**:
  - **Single best metric for recomposition success**
  - Shows you are not just losing weight, but changing composition
  - Ratios near 1:1 are extremely difficult to achieve
  - Indicates optimal training and nutrition balance

---

### 2️⃣ **Plateau Alerts - Automated Progress Monitoring**

Dynamic alert system that detects when progress stalls.

#### 🚨 Alert Levels
- **CRITICAL** (Red): Plateaus in 2+ metrics for 30 days
- **WARNING** (Orange): Plateaus in multiple metrics for 21 days
- **CAUTION** (Yellow): Single metric plateau for 14-21 days

#### What It Detects
- **Muscle Plateaus**: No growth in 14/21/30 days
- **Fat Plateaus**: No fat loss in 14/21/30 days
- **Weight Plateaus**: No change in total weight

#### Why It Matters
- **Adaptive Training**: Your body adapts to stimulus - plateaus signal need for change
- **Prevents Spinning Wheels**: Alerts you before wasting months on ineffective approach
- **Evidence-Based Adjustments**: Suggests specific actions:
  - Increase training volume/intensity
  - Adjust caloric intake
  - Modify macro ratios
  - Implement deload week
  - Check recovery quality

#### How to Use
- Review alerts weekly
- If plateau detected, analyze:
  - Training logs (are you progressing in lifts?)
  - Nutrition tracking (hitting protein targets?)
  - Sleep quality (7-9 hours?)
  - Stress levels (cortisol impacts body comp)

---

### 3️⃣ **Nutrition Tracking - Fuel Your Progress**

Track your daily nutrition intake to understand how your diet impacts body composition.

#### What You Can Track
- **Daily Calorie Intake**: Total calories consumed per day
- **Macronutrient Split**: Protein, Fat, and Carbs percentages (must total 100%)
- **Automatic Calculations**: Grams of each macro based on your calorie intake

#### Insights Provided
**💪 Protein Intake Analysis**
- **Grams per kg body weight**: Optimal range is 1.6-2.2 g/kg
- **Status Assessment**: EXCELLENT, GOOD, MODERATE, or LOW
- **Recommendations**: Personalized suggestions based on your current intake

**🔥 Training Phase Detection**
- **TDEE Estimation**: Reverse-calculated from your actual weight changes
- **Caloric Balance**: Surplus or deficit calculation
- **Phase Classification**:
  - MAINTENANCE: Stable weight (±0.1kg/week)
  - LEAN BULK: +0.2 to +0.5kg/week (optimal muscle building)
  - AGGRESSIVE BULK: +0.5kg/week (high fat gain risk)
  - RECOMPOSITION: -0.3 to +0.2kg/week (body recomp zone)
  - CUT: -0.3 to -0.8kg/week (fat loss phase)
  - AGGRESSIVE CUT: -0.8kg/week+ (muscle loss risk)

**⚡ Nutrient Partitioning Efficiency**
- **Recomp Score**: (Muscle Gained - Fat Gained) ÷ Caloric Balance × 100
- **Quality Ratings**: ELITE (>15), EXCELLENT (>10), GOOD (>5)
- **Muscle per 100 cal**: How efficiently you convert surplus calories to muscle

#### How to Use
1. **Set Your Intake**: Enter current daily calories and macro split
2. **Save Settings**: Persists across sessions
3. **Monitor Insights**: Check if your intake aligns with your goals
4. **Adjust as Needed**: Use TDEE and phase detection to optimize intake

---

### 4️⃣ **Goal Tracking - Predict Your Success**

Set target weight goals and get science-based predictions for achieving them.

#### What You Can Set
- **Target Weight**: Your desired end weight in kg
- **Target Date** (Optional): When you want to reach your goal
- **Clear Goal**: Remove goal when achieved or changing direction

#### Predictions Provided
**🎯 Goal Progress**
- Current → Target weight visualization
- Total weight to gain or lose

**📅 Time to Goal**
- **Weeks at Current Pace**: Based on your last 30 days
- **ETA Date**: Estimated arrival at your goal weight
- **Progress Warnings**: Alerts if moving away from goal or insufficient progress

**🔥 Recommended Calorie Intake**
- **Optimal Daily Calories**: Calculated for healthy rate of change
- **Adjustment from TDEE**: How much to increase/decrease
- **Optimal Rate Guidance**:
  - Cutting: 0.5-0.8 kg/week (preserves muscle)
  - Bulking: 0.2-0.5 kg/week (minimizes fat gain)

**📊 Body Composition Predictions**
- **Predicted Muscle at Goal**: Based on current trajectory
- **Predicted Fat at Goal**: Using recent composition changes
- **Quality Assessment**:
  - **ELITE/EXCELLENT** (Cutting): Preserving muscle while losing fat
  - **ELITE/EXCELLENT** (Bulking): Gaining 2:1 or better muscle:fat ratio
  - **WARNING**: Risk of excessive muscle loss or fat gain

#### Example Predictions
If you set target weight of **70kg** (from current 66.6kg):
- **Goal**: 66.6kg → 70kg (3.4kg to gain)
- **At Current Pace**: ~20 weeks
- **Recommended Intake**: 3300-3400 cal/day
- **Prediction**: Muscle +3.0kg, Fat +0.8kg - EXCELLENT trajectory

#### How to Use
1. **Set Your Goal**: Enter target weight and optional date
2. **Review Predictions**: Check if timeline and composition changes are acceptable
3. **Adjust Strategy**: Use recommended calories to optimize your approach
4. **Track Progress**: Monitor how actual progress compares to predictions
5. **Iterate**: Update goal as you progress or when priorities change

---

### 5️⃣ **Main Composition Trends - The Big Picture**

Multi-line chart showing muscle, fat, and weight over time with 7-day moving averages.

#### How to Read
- **Green line**: Muscle mass trajectory
- **Purple line**: Fat mass trajectory  
- **Navy line**: Total body weight
- **Dotted lines**: 7-day moving averages (smooths daily fluctuations)

#### Ideal Patterns
1. **Classic Recomp**: Green trending up, Purple trending down, Navy relatively flat
2. **Lean Bulk**: All three trending up, but green > purple slope
3. **Cut**: All trending down, but purple slope > green slope

#### What to Look For
- **Divergence**: Muscle and fat lines moving apart = successful recomp
- **Parallel lines**: Both increasing/decreasing = need strategy adjustment
- **Volatility**: Large day-to-day swings = water weight, glycogen, measurement error
- **7-day MA**: Use this for trend analysis, ignore daily noise

#### Why It Matters
- **Visual Progress**: Sometimes you don't "feel" different but the trend is clear
- **Motivation**: Seeing lines diverge is incredibly motivating
- **Course Correction**: Identify when approach stops working

---

### 6️⃣ **Recomposition Divergence Chart - Dual-Axis Victory**

Shows muscle (left axis) and fat (right axis) on same chart with opposite scales.

#### How to Read
- **Green bars going UP**: Muscle increasing (good!)
- **Purple bars going DOWN**: Fat decreasing (good!)
- When both happen simultaneously = **recomposition nirvana**

#### Why Separate Axes
- Different scales (you might have 65kg muscle but only 17kg fat)
- Visualizes the divergence more dramatically
- Makes the "pull apart" effect obvious

#### Psychological Impact
- More dramatic than overlaid lines
- Clearly shows you're not just "losing weight" but transforming composition
- Great for sharing progress with coaches/community

---

### 7️⃣ **Segment Analysis Charts - Find Your Weak Points**

Six charts showing muscle and fat distribution across body segments.

#### Trunk Analysis
- **What it shows**: Core muscle vs. fat
- **Why it matters**: 
  - Trunk fat correlates with visceral fat (dangerous)
  - Core strength impacts all other lifts
  - Postural stability and injury prevention

#### Leg Analysis (Left, Right, Combined)
- **What it shows**: Quadriceps, hamstrings, glutes, calves
- **Why it matters**:
  - Legs = largest muscle group = biggest metabolic impact
  - Bilateral comparisons detect imbalances
  - Imbalances increase injury risk
  - Combined legs should show highest muscle mass

#### Arm Analysis (Left, Right)
- **What it shows**: Biceps, triceps, forearms
- **Why it matters**:
  - Arm symmetry for aesthetics and function
  - Detect dominance issues (e.g., right arm stronger)
  - Smaller muscle group but visible progress

#### How to Use
1. **Identify Imbalances**: Left/right differences > 5% = focus needed
2. **Target Weak Points**: Lagging segments need volume/frequency increase
3. **Injury Prevention**: Asymmetry correlates with injury risk
4. **Track Specific Goals**: Want bigger legs? Track that segment specifically

---

### 8️⃣ **Health Indicators - Beyond Aesthetics**

Three critical health metrics tracked over time.

#### 🫀 Visceral Fat Rating (Scale: 1-59)
- **What it is**: Fat surrounding internal organs
- **Ranges**:
  - 1-12: Healthy
  - 13-59: Excessive (health risk)
- **Why it matters**:
  - **Most dangerous type of fat**
  - Increases risk: Heart disease, diabetes, stroke, cancer
  - Metabolically active (secretes inflammatory hormones)
  - Cannot see it (subcutaneous fat is visible, visceral is hidden)
- **Target**: < 10 is excellent, < 13 is acceptable

#### 🧬 Metabolic Age (Years)
- **What it is**: Age your metabolism functions like
- **How it's calculated**: BMR compared to population averages
- **Why it matters**:
  - Lower than actual age = efficient metabolism
  - Indicates overall metabolic health
  - Motivation metric (54 years old but metabolically 38!)
- **Target**: Equal to or less than chronological age

#### 📊 BMI (Body Mass Index)
- **What it is**: Weight (kg) / Height (m)²
- **Standard ranges**:
  - < 18.5: Underweight
  - 18.5-24.9: Normal
  - 25-29.9: Overweight
  - 30+: Obese
- **Critical limitation for recomp**:
  - **Does NOT account for muscle mass**
  - Muscular individuals often "overweight" by BMI
  - Bodybuilders frequently "obese" by BMI
  - Use body composition metrics instead
- **Why it's included**: Reference point, medical relevance

---

### 9️⃣ **Phase Angle Chart - Recovery & Cellular Health**

Advanced bioelectrical impedance metric tracking cellular integrity.

#### What Phase Angle Measures
- Electrical properties of cell membranes
- Intracellular vs. extracellular water distribution
- Cell membrane health and integrity
- Overall cellular function

#### How to Read
- **> 5.0°**: Excellent cellular health
- **4.5-5.0°**: Good
- **4.0-4.5°**: Fair
- **< 4.0°**: Poor (potential overtraining, illness, aging)

#### Why It Matters for Recomp
- **Recovery Indicator**: Low phase angle = inadequate recovery
- **Overtraining Detection**: Drops significantly when overtrained
- **Nutrition Status**: Improves with adequate protein and hydration
- **Training Readiness**: High phase angle = ready for hard training
- **Aging Biomarker**: Generally declines with age, training can maintain it

#### How to Use
- **Track trends** more than absolute values
- **Drops > 0.3° in a week**: Red flag - check recovery
- **Consistently low**: Evaluate training volume, sleep, stress, nutrition
- **Consistently high**: Sign of good training/recovery balance

---

### 🔟 **Body Composition Ratios - Professional Metrics**

Five advanced metrics used by sports scientists and elite coaches.

#### Fat-Free Mass Percentage (FFM%)
- **Formula**: (Muscle + Bone + Water) / Total Weight × 100
- **Ranges**:
  - Men: 75-85% = athletic, > 85% = elite
  - Women: 65-75% = athletic, > 75% = elite
- **Why it matters**:
  - Inverse of body fat percentage
  - More precise than just tracking weight
  - Gold standard for body composition assessment

#### Muscle to Bone Ratio
- **Formula**: Muscle Mass / Bone Mass
- **Typical range**: 12-18:1
- **Why it matters**:
  - Structural health indicator
  - Bone supports muscle (need balance)
  - Very low = inadequate bone density for muscle mass
  - Tracks skeletal health alongside muscle

#### BMR per kg Muscle (Metabolic Efficiency)
- **Formula**: Basal Metabolic Rate / Muscle Mass
- **What it shows**: How metabolically active your muscle is
- **Why it matters**:
  - Higher = more calories burned per kg muscle
  - Training increases muscle metabolic activity
  - Tracks training-induced metabolic adaptations

#### Muscle Quality Index
- **Formula**: Muscle Mass / Height²
- **Similar to**: BMI but for muscle only
- **Why it matters**:
  - Normalizes muscle mass for height
  - Compare progress over time
  - Benchmark against population (harder to find norms)

#### Hydration Ratio
- **Formula**: Intracellular Water / Extracellular Water
- **Optimal range**: 1.2-1.5
- **Why it matters**:
  - > 1.0 = more water inside cells (good)
  - < 1.0 = more water outside cells (edema, inflammation)
  - Indicator of cellular health and hydration status

---

### 1️⃣1️⃣ **Recovery Dashboard - Train Smarter, Not Just Harder**

#### Training Readiness Indicator
- **Composite score**: 0-100 based on Phase Angle, Hydration, BMR
- **Color codes**:
  - 🟢 90-100: READY - Push hard, heavy weights, high volume
  - 🟡 80-89: GOOD - Normal training, avoid PRs
  - 🟠 70-79: MODERATE - Reduce volume 20-30%
  - 🔴 60-69: CAUTION - Light training or active recovery
  - ⚫ < 60: REST - Take a day off

#### Recovery Score Chart (0-100 scale)
- **Components**:
  - **Phase Angle** (40 points): Cellular health
  - **Hydration Ratio** (30 points): Water distribution
  - **Metabolic Efficiency** (30 points): BMR per kg muscle

#### Why This Matters
- **Prevents Overtraining**: Tells you when to back off
- **Optimizes Gains**: Hard training during high readiness = max adaptation
- **Reduces Injury Risk**: Training on low recovery = injury prone
- **Personalized**: Uses YOUR baseline, not population averages

#### How to Use
- **Pre-workout check**: Adjust session based on score
- **Weekly trends**: Consistently low = overreaching
- **Peaking for events**: Schedule hard sessions on high-score days
- **Deload timing**: Series of low scores = time for deload week

---

### 🔟 **Segmental Phase Angle - Regional Recovery**

Phase angle breakdown for trunk, arms, and legs.

#### Why Measure by Segment
- **Detect Regional Overtraining**: Legs low after hard squat week
- **Balance Training Load**: If arm phase angle drops, reduce upper volume
- **Injury Prevention**: Low segment phase angle before injury
- **Return to Training**: Post-injury, watch affected segment recovery

#### Practical Applications
- **Monday: Leg day planned, but leg phase angle dropped** → Switch to upper body
- **Arm phase angle consistently lagging** → Reduce arm volume 20%
- **Trunk phase angle excellent** → Can add core-intensive work

---

### 1️⃣2️⃣ **Water Distribution Chart - Hydration & Inflammation**

Tracks intracellular vs. extracellular water over time.

#### Ideal Pattern
- **Intracellular (purple) line**: Higher and stable
- **Extracellular (teal) line**: Lower and stable
- **Gap between lines**: Should be consistent

#### Warning Signs
- **Extracellular spiking**: 
  - Inflammation
  - High sodium intake
  - Stress/cortisol
  - Inadequate sleep
- **Intracellular dropping**:
  - Dehydration
  - Muscle glycogen depletion
  - Overtraining

#### Why It Matters
- **True hydration status**: Not just "drink more water"
- **Inflammation tracking**: Chronic inflammation kills gains
- **Glycogen assessment**: Low intracellular = depleted glycogen stores
- **Recovery indicator**: Proper ratio = recovered

---

### 1️⃣3️⃣ **Velocity & Acceleration Metrics - Rate of Change**

Shows HOW FAST you're changing, not just the total change.

#### Current Velocity (kg/week)
- **Muscle velocity**: +0.25 kg/week = excellent natural rate
- **Fat velocity**: -0.5 kg/week = sustainable fat loss
- **Weight velocity**: Near zero = successful recomp

#### Acceleration (kg/week²)
- **Positive acceleration**: Rate of change is increasing (compounding gains)
- **Negative acceleration**: Rate slowing down (plateau approaching)
- **Zero acceleration**: Steady linear progress (ideal)

#### Why It Matters
- **Early plateau detection**: Negative acceleration predicts plateau
- **Progress validation**: Confirms you're still making gains
- **Timeline estimation**: Predict when you'll hit goals

---

### 1️⃣4️⃣ **90-Day Projection Charts - Crystal Ball**

Three forecast charts using linear regression to project future values.

#### Muscle Mass Projection
- **Shows**: Where muscle mass will be in 90 days if current trend continues
- **Confidence interval**: Shaded area = 95% likely to land in this range
- **R² score**: How confident the model is (> 0.7 = reliable)

#### Fat Mass Projection  
- **Shows**: Predicted fat mass in 90 days
- **Use case**: "At this rate, I'll lose 5kg fat by summer"

#### Body Weight Projection
- **Shows**: Total weight trajectory
- **Validation**: If recomping correctly, projection should be flat

#### How to Use
- **Goal setting**: "I want 68kg muscle - will I get there in 90 days?"
- **Strategy validation**: If projection doesn't align with goals, adjust NOW
- **Motivation**: Visual proof that consistency pays off
- **Reality check**: "Losing 20kg in 90 days" is unrealistic - see the math

#### Important Caveats
- **Linear assumption**: Assumes current rate continues (rarely perfect)
- **Not prophesy**: Life happens, rates change
- **Use for direction**: Is trend going the right way?

---

### 1️⃣5️⃣ **Scan History Table - The Raw Data**

Sortable, filterable table of all scan data.

#### How to Use
- **Sort by date**: See chronological progress
- **Filter by metric**: Find specific measurements
- **Export**: Copy data for personal tracking
- **Spot check**: Verify unusual spikes in charts

#### Key Columns
- **Date**: Scan timestamp
- **All 34 metrics**: Complete Boditrax output
- **Calculated fields**: Ratios, averages, derived metrics

---

## 🎯 How to Use This Dashboard for Maximum Results

### Weekly Review Protocol
1. **Check KPI cards**: Am I progressing in muscle/fat divergence?
2. **Review plateau alerts**: Any stalls detected?
3. **Analyze trends**: Are lines moving in the right direction?
4. **Check recovery score**: Training too hard or not enough?
5. **Review projections**: On track for 90-day goals?

### Monthly Deep Dive
1. **Segment analysis**: Any weak points or imbalances?
2. **Health indicators**: Visceral fat, metabolic age improving?
3. **Body composition ratios**: FFM% increasing?
4. **Velocity metrics**: Progress accelerating or slowing?

### When to Adjust Strategy

#### Increase Training Volume If:
- Muscle velocity < 0.1 kg/week for 4+ weeks
- Muscle mass plateaued
- Recovery score consistently > 85

#### Reduce Training Volume If:
- Recovery score consistently < 70
- Phase angle dropping > 0.5° in 2 weeks
- Muscle velocity negative (losing muscle)

#### Adjust Nutrition If:
- Fat velocity near zero (not losing fat)
- Muscle velocity negative (losing muscle = eating too little)
- Weight velocity > ±0.5 kg/week (too aggressive)

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/marcus-furius/boditrax_composition_dashboard.git
cd boditrax_composition_dashboard

# Install dependencies
pip install -r requirements.txt

# Add your Boditrax CSV file
# Place your CSV export in the data/ folder

# Run dashboard
python app_complete.py

# Open browser to http://localhost:8052
```

### First Time Setup

1. **Export your Boditrax data**:
   - Log into Boditrax account
   - Export all scans as CSV
   - Save to `data/` folder

2. **Upload via dashboard**:
   - Click "📂 Select Boditrax CSV File"
   - Choose your file
   - Refresh page to load data

3. **Explore visualizations**:
   - Use date range filters to focus on specific periods
   - Click plateau alerts to see recommendations
   - Check recovery score before workouts

---

## 📈 Interpreting Your Results

### Success Patterns

#### Elite Recomposition (Rare)
- Muscle: +0.3-0.5 kg/week
- Fat: -0.4-0.6 kg/week  
- Weight: ±0.1 kg/week
- Recomp ratio: 0.8-1.0

#### Excellent Recomposition
- Muscle: +0.2-0.3 kg/week
- Fat: -0.3-0.5 kg/week
- Weight: -0.1-0.2 kg/week
- Recomp ratio: 0.6-0.8

#### Good Lean Bulk
- Muscle: +0.3-0.5 kg/week
- Fat: +0.1-0.2 kg/week
- Weight: +0.4-0.7 kg/week
- Recomp ratio: 2.5+ (more muscle than fat)

#### Good Cut
- Muscle: -0.0-0.1 kg/week (minimal loss)
- Fat: -0.5-0.8 kg/week
- Weight: -0.5-0.9 kg/week
- Recomp ratio: 0.1-0.3 (preserving muscle)

### Red Flags

#### Spinning Your Wheels
- All metrics flat for 30+ days
- Weight stable but no composition change
- → **Action**: Increase training volume or adjust nutrition

#### Skinny Fat Trajectory
- Losing muscle and fat proportionally
- Low recomp ratio during cut
- → **Action**: Increase protein, add resistance training

#### Dirty Bulk
- Gaining fat faster than muscle
- Visceral fat increasing
- → **Action**: Reduce caloric surplus, increase cardio

---

## 🧪 Science Behind the Metrics

### Phase Angle Research
- Validated predictor of nutritional status
- Correlates with muscle quality and strength
- Used in clinical settings for health assessment
- Sports science: Tracks training adaptation

### Body Composition Importance
- Total weight ≠ health or fitness
- Muscle mass correlates with longevity
- Visceral fat = #1 modifiable health risk factor
- Body recomposition possible at any age (even 40+)

### Recovery Science
- Overtraining = negative progress
- Optimal stimulus: Hard training + adequate recovery
- HRV, phase angle, hydration = recovery markers
- Personalized recovery needs vary widely

---

## 🎓 Resources

### Recommended Reading
- "The Renaissance Diet 2.0" - Mike Israetel (nutrition periodization)
- "Scientific Principles of Strength Training" - Israetel et al.
- "Muscle and Strength Pyramids" - Eric Helms

### Research
- [Body Composition and Health Outcomes](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6315740/)
- [Phase Angle and Athletic Performance](https://pubmed.ncbi.nlm.nih.gov/31234028/)
- [Recomposition in Resistance-Trained Athletes](https://jissn.biomedcentral.com/articles/10.1186/s12970-020-00396-1)

---

## ⚙️ Features

- 📊 **15 Interactive Visualizations**
- 🎯 **Automated Plateau Detection**
- 💪 **Body Composition Ratios**
- 🧬 **Recovery Scoring System**
- 🔮 **90-Day Predictive Analytics**
- 📁 **File Upload with Config Persistence**
- 📄 **PDF/HTML Export**
- 📅 **Date Range Filtering**
- 📈 **7-Day Moving Averages**
- 🎨 **Jazz Pharmaceuticals Branded Theme**

---

## 🤝 Contributing

Found a bug? Have feature suggestions? 

- Open an issue on GitHub
- Submit pull requests
- Share your success stories!

---

## 📜 License

This project is open source and available for personal use.

---

## 💪 Your Recomposition Journey

Remember: **Body recomposition is a marathon, not a sprint.**

- Trust the process
- Track consistently  
- Adjust based on data, not feelings
- Celebrate small wins (0.5kg muscle = huge!)
- Focus on long-term trends, not daily fluctuations

**This dashboard gives you the data. Now go build the physique.**

---

🤖 *Built with Claude Code - https://claude.com/claude-code*

Co-Authored-By: Claude <noreply@anthropic.com>
