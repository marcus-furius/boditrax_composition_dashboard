"""
Body Recomposition Dashboard - Complete Version
Combines all features from both original and enhanced versions
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import json
import os
import base64
import io

# ============================================================================
# CONFIG FILE HANDLING
# ============================================================================

CONFIG_FILE = 'config.json'

def load_config():
    """Load configuration from JSON file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {'last_file': 'data/BoditraxAccount_20251006_102227.csv'}

def save_config(config):
    """Save configuration to JSON file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# ============================================================================
# DATA LOADING MODULE
# ============================================================================

def load_boditrax_data(filepath='data/BoditraxAccount_20251006_102227.csv'):
    """Parse multi-section Boditrax CSV export"""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    scan_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'User Scan Details':
            scan_start = i + 2
            break

    if scan_start is None:
        raise ValueError("Could not find 'User Scan Details' section in CSV")

    scan_data = []
    for i in range(scan_start, len(lines)):
        line = lines[i].strip()
        if not line or line == 'User Login Details':
            break

        parts = line.split(',')
        if len(parts) == 3:
            date_str = parts[2].replace('\u202f', ' ')
            scan_data.append({
                'Metric': parts[0],
                'Value': parts[1],
                'Date': date_str
            })

    df = pd.DataFrame(scan_data)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Date'] = df['Date'].apply(lambda x: date_parser.parse(x))

    return df


def transform_to_wide_format(df):
    """Pivot long-format data to wide format"""
    df_wide = df.pivot_table(
        index='Date',
        columns='Metric',
        values='Value',
        aggfunc='first'
    )
    df_wide = df_wide.sort_index()
    return df_wide


def add_moving_averages(df_wide, window=7):
    """Add moving averages for key metrics"""
    df_ma = df_wide.copy()
    key_metrics = ['MuscleMass', 'FatMass', 'BodyWeight', 'VisceralFatRating',
                   'MetabolicAge', 'BasalMetabolicRatekJ']

    for metric in key_metrics:
        if metric in df_ma.columns:
            df_ma[f'{metric}_MA{window}'] = df_ma[metric].rolling(window=window, min_periods=1).mean()

    return df_ma


def calculate_phase_angle_average(df_wide):
    """Calculate average phase angle from all body segments"""
    phase_angle_cols = [col for col in df_wide.columns if 'PhaseAngle' in col]
    if phase_angle_cols:
        df_wide['PhaseAngle_Avg'] = df_wide[phase_angle_cols].abs().mean(axis=1)
    return df_wide


def calculate_body_composition_ratios(df_wide):
    """Calculate advanced body composition ratios and efficiency metrics"""
    df_ratios = df_wide.copy()

    # 1. Fat-Free Mass Percentage (gold standard for body composition)
    if 'FatFreeMass' in df_ratios.columns and 'BodyWeight' in df_ratios.columns:
        df_ratios['FatFreeMass_Pct'] = (df_ratios['FatFreeMass'] / df_ratios['BodyWeight']) * 100

    # 2. Muscle to Bone Ratio (structural health indicator)
    if 'MuscleMass' in df_ratios.columns and 'BoneMass' in df_ratios.columns:
        df_ratios['MuscleBone_Ratio'] = df_ratios['MuscleMass'] / df_ratios['BoneMass']

    # 3. BMR per kg Muscle (metabolic efficiency)
    if 'BasalMetabolicRatekJ' in df_ratios.columns and 'MuscleMass' in df_ratios.columns:
        df_ratios['BMR_per_kg_Muscle'] = df_ratios['BasalMetabolicRatekJ'] / df_ratios['MuscleMass']

    # 4. Muscle Quality Index (like BMI but for muscle - strength potential)
    if 'MuscleMass' in df_ratios.columns and 'Height' in df_ratios.columns:
        height_m = df_ratios['Height'] / 100  # Convert cm to meters
        df_ratios['Muscle_Quality_Index'] = df_ratios['MuscleMass'] / (height_m ** 2)

    # 5. Hydration Ratio (cellular health - intracellular vs extracellular water)
    if 'IntraCellularWaterMass' in df_ratios.columns and 'ExtraCellularWaterMass' in df_ratios.columns:
        df_ratios['Hydration_Ratio'] = df_ratios['IntraCellularWaterMass'] / df_ratios['ExtraCellularWaterMass']

    return df_ratios


# ============================================================================
# METRICS CALCULATION MODULE
# ============================================================================

def detect_plateaus(df_wide, plateau_threshold=0.5):
    """
    Detect plateaus in key metrics over 14, 21, and 30 day windows
    A plateau is defined as < 0.5% change over the period

    Returns dict with plateau status and recommendations
    """
    plateaus = {
        'muscle_14d': False, 'muscle_21d': False, 'muscle_30d': False,
        'fat_14d': False, 'fat_21d': False, 'fat_30d': False,
        'weight_14d': False, 'weight_21d': False, 'weight_30d': False,
        'alerts': [],
        'severity': 'NONE'  # NONE, LOW, MEDIUM, HIGH
    }

    if len(df_wide) < 3:
        return plateaus

    latest_date = df_wide.index[-1]

    # Check each time window
    for days, suffix in [(14, '14d'), (21, '21d'), (30, '30d')]:
        window_start = latest_date - timedelta(days=days)
        window_data = df_wide[df_wide.index >= window_start]

        if len(window_data) < 2:
            continue

        first_val = window_data.iloc[0]
        last_val = window_data.iloc[-1]

        # Check Muscle Mass
        if 'MuscleMass' in window_data.columns:
            muscle_pct_change = abs((last_val['MuscleMass'] - first_val['MuscleMass']) / first_val['MuscleMass'] * 100)
            if muscle_pct_change < plateau_threshold:
                plateaus[f'muscle_{suffix}'] = True

        # Check Fat Mass (we want change here, but stalling is also a plateau)
        if 'FatMass' in window_data.columns:
            fat_pct_change = abs((last_val['FatMass'] - first_val['FatMass']) / first_val['FatMass'] * 100)
            if fat_pct_change < plateau_threshold:
                plateaus[f'fat_{suffix}'] = True

        # Check Body Weight
        if 'BodyWeight' in window_data.columns:
            weight_pct_change = abs((last_val['BodyWeight'] - first_val['BodyWeight']) / first_val['BodyWeight'] * 100)
            if weight_pct_change < plateau_threshold:
                plateaus[f'weight_{suffix}'] = True

    # Generate alerts based on detected plateaus
    alerts = []

    # Muscle plateaus
    if plateaus['muscle_30d']:
        alerts.append({
            'metric': 'Muscle Mass',
            'severity': 'HIGH',
            'message': '⚠️ MUSCLE GROWTH STALLED (30+ days)',
            'recommendation': 'Consider: Progressive overload, increase training volume, or calorie surplus'
        })
        plateaus['severity'] = 'HIGH'
    elif plateaus['muscle_21d']:
        alerts.append({
            'metric': 'Muscle Mass',
            'severity': 'MEDIUM',
            'message': '⚠️ Muscle plateau detected (21 days)',
            'recommendation': 'Adjust training stimulus or check protein intake'
        })
        if plateaus['severity'] == 'NONE':
            plateaus['severity'] = 'MEDIUM'
    elif plateaus['muscle_14d']:
        alerts.append({
            'metric': 'Muscle Mass',
            'severity': 'LOW',
            'message': '⚠️ Muscle gain slowing (14 days)',
            'recommendation': 'Monitor closely - may need intervention soon'
        })
        if plateaus['severity'] == 'NONE':
            plateaus['severity'] = 'LOW'

    # Fat loss plateaus
    if plateaus['fat_30d']:
        alerts.append({
            'metric': 'Fat Mass',
            'severity': 'HIGH',
            'message': '⚠️ FAT LOSS STALLED (30+ days)',
            'recommendation': 'Consider: Increase cardio, reduce calories, or refeed strategy'
        })
        if plateaus['severity'] != 'HIGH':
            plateaus['severity'] = 'HIGH'
    elif plateaus['fat_21d']:
        alerts.append({
            'metric': 'Fat Mass',
            'severity': 'MEDIUM',
            'message': '⚠️ Fat loss plateau detected (21 days)',
            'recommendation': 'Adjust calorie deficit or activity level'
        })
        if plateaus['severity'] == 'NONE':
            plateaus['severity'] = 'MEDIUM'
    elif plateaus['fat_14d']:
        alerts.append({
            'metric': 'Fat Mass',
            'severity': 'LOW',
            'message': '⚠️ Fat loss slowing (14 days)',
            'recommendation': 'Watch trend - may stabilize naturally'
        })
        if plateaus['severity'] == 'NONE':
            plateaus['severity'] = 'LOW'

    # Weight plateaus (overall recomp context)
    if plateaus['weight_30d'] and not plateaus['muscle_30d'] and not plateaus['fat_30d']:
        # Weight stable but composition changing = good recomp
        alerts.append({
            'metric': 'Body Weight',
            'severity': 'INFO',
            'message': '✅ Weight stable with body recomposition',
            'recommendation': 'Excellent - maintaining weight while changing composition'
        })

    plateaus['alerts'] = alerts

    return plateaus


def calculate_recomp_metrics(df_wide):
    """Calculate body recomposition metrics and deltas"""
    first_scan = df_wide.iloc[0]
    latest_scan = df_wide.iloc[-1]
    prev_week_scan = df_wide.iloc[-2] if len(df_wide) > 1 else first_scan

    muscle_change = latest_scan['MuscleMass'] - first_scan['MuscleMass']
    muscle_pct_change = (muscle_change / first_scan['MuscleMass']) * 100
    muscle_wow = latest_scan['MuscleMass'] - prev_week_scan['MuscleMass']
    muscle_wow_pct = (muscle_wow / prev_week_scan['MuscleMass']) * 100

    fat_change = latest_scan['FatMass'] - first_scan['FatMass']
    fat_pct_change = (fat_change / first_scan['FatMass']) * 100
    fat_wow = latest_scan['FatMass'] - prev_week_scan['FatMass']
    fat_wow_pct = (fat_wow / prev_week_scan['FatMass']) * 100

    weight_change = latest_scan['BodyWeight'] - first_scan['BodyWeight']
    weight_pct_change = (weight_change / first_scan['BodyWeight']) * 100
    weight_wow = latest_scan['BodyWeight'] - prev_week_scan['BodyWeight']

    if fat_change < 0:
        recomp_ratio = abs(muscle_change / fat_change) if fat_change != 0 else 0
        recomp_status = "WINNING" if muscle_change > 0 else "LOSING MUSCLE"
    else:
        recomp_ratio = 0
        recomp_status = "GAINING FAT"

    return {
        'muscle_change': muscle_change,
        'muscle_pct_change': muscle_pct_change,
        'muscle_wow': muscle_wow,
        'muscle_wow_pct': muscle_wow_pct,
        'fat_change': fat_change,
        'fat_pct_change': fat_pct_change,
        'fat_wow': fat_wow,
        'fat_wow_pct': fat_wow_pct,
        'weight_change': weight_change,
        'weight_pct_change': weight_pct_change,
        'weight_wow': weight_wow,
        'recomp_ratio': recomp_ratio,
        'recomp_status': recomp_status,
        'latest_visceral_fat': latest_scan['VisceralFatRating'],
        'latest_metabolic_age': latest_scan['MetabolicAge'],
        'latest_actual_age': latest_scan['Age'],
        'latest_bmr': latest_scan['BasalMetabolicRatekJ'],
        'latest_boditrax_score': latest_scan['BoditraxScore'],
        'latest_phase_angle': latest_scan.get('PhaseAngle_Avg', 0),
        'first_scan_date': df_wide.index[0],
        'latest_scan_date': df_wide.index[-1],
        'total_scans': len(df_wide)
    }


def calculate_recovery_score(df_wide):
    """
    Calculate comprehensive recovery score combining:
    - Phase Angle (cellular health & recovery)
    - Hydration Ratio (intracellular vs extracellular water)
    - BMR per kg Muscle (metabolic efficiency)

    Uses RELATIVE scoring based on YOUR personal baseline range
    Returns df with RecoveryScore (0-100) and ReadinessStatus
    """
    df_recovery = df_wide.copy()

    # Component 1: Phase Angle Score (0-40 points) - Relative to YOUR range
    if 'PhaseAngle_Avg' in df_recovery.columns:
        phase_angle = df_recovery['PhaseAngle_Avg'].abs()

        # Calculate personal baseline (median of all data)
        pa_baseline = phase_angle.median()
        pa_std = phase_angle.std()

        # Score based on deviation from YOUR baseline
        # +1 std or more = 40 pts, baseline = 30 pts, -1 std or more = 20 pts
        def score_phase_angle(x):
            if pd.isna(x):
                return 30
            deviation = (x - pa_baseline) / pa_std if pa_std > 0 else 0
            if deviation >= 1.0:
                return 40  # Well above your average
            elif deviation >= 0.5:
                return 37
            elif deviation >= 0:
                return 33
            elif deviation >= -0.5:
                return 27
            elif deviation >= -1.0:
                return 23
            else:
                return 20  # Well below your average

        df_recovery['PhaseAngle_Score'] = phase_angle.apply(score_phase_angle)
    else:
        df_recovery['PhaseAngle_Score'] = 30

    # Component 2: Hydration Ratio Score (0-30 points) - Relative to YOUR range
    if 'Hydration_Ratio' in df_recovery.columns:
        hydration = df_recovery['Hydration_Ratio']

        # Calculate personal baseline
        h_baseline = hydration.median()
        h_std = hydration.std()

        # Score based on deviation (higher is better, but extreme values are bad)
        def score_hydration(x):
            if pd.isna(x):
                return 25
            deviation = (x - h_baseline) / h_std if h_std > 0 else 0
            # Optimal is near baseline, too high or too low is bad
            abs_dev = abs(deviation)
            if abs_dev <= 0.3:
                return 30  # Close to your average
            elif abs_dev <= 0.7:
                return 27
            elif abs_dev <= 1.2:
                return 23
            elif abs_dev <= 1.8:
                return 18
            else:
                return 15  # Far from your normal

        df_recovery['Hydration_Score'] = hydration.apply(score_hydration)
    else:
        df_recovery['Hydration_Score'] = 25

    # Component 3: Metabolic Efficiency Score (0-30 points) - 30-day rolling baseline
    if 'BMR_per_kg_Muscle' in df_recovery.columns:
        bmr_efficiency = df_recovery['BMR_per_kg_Muscle']

        # Calculate 30-day baseline (longer term trend)
        bmr_baseline = bmr_efficiency.rolling(window=30, min_periods=7).median()
        bmr_std = bmr_efficiency.rolling(window=30, min_periods=7).std()

        # Score based on deviation from 30-day baseline
        def score_metabolic(current, baseline, std):
            if pd.isna(current) or pd.isna(baseline) or pd.isna(std) or std == 0:
                return 25
            deviation = (current - baseline) / std
            # Stable is good, large changes indicate stress
            abs_dev = abs(deviation)
            if abs_dev <= 0.3:
                return 30  # Very stable
            elif abs_dev <= 0.7:
                return 27  # Slight variation
            elif abs_dev <= 1.2:
                return 23  # Moderate variation
            elif abs_dev <= 2.0:
                return 18  # High variation
            else:
                return 15  # Extreme variation (stress/overtraining)

        df_recovery['Metabolic_Score'] = df_recovery.apply(
            lambda row: score_metabolic(
                row['BMR_per_kg_Muscle'],
                bmr_baseline.loc[row.name] if row.name in bmr_baseline.index else pd.NA,
                bmr_std.loc[row.name] if row.name in bmr_std.index else pd.NA
            ), axis=1
        )
    else:
        df_recovery['Metabolic_Score'] = 25

    # Total Recovery Score (0-100)
    df_recovery['Recovery_Score'] = (
        df_recovery['PhaseAngle_Score'] +
        df_recovery['Hydration_Score'] +
        df_recovery['Metabolic_Score']
    )

    # Readiness Status based on total score
    df_recovery['Readiness_Status'] = df_recovery['Recovery_Score'].apply(lambda x:
        'READY' if x >= 90 else
        'GOOD' if x >= 80 else
        'MODERATE' if x >= 70 else
        'CAUTION' if x >= 60 else
        'REST NEEDED'
    )

    return df_recovery


def calculate_velocity_metrics(df_wide):
    """
    Calculate velocity (rate of change) and acceleration for key metrics
    Returns current velocity in kg/week and acceleration in kg/week²
    """
    if len(df_wide) < 3:
        return {}

    # Calculate time deltas in weeks
    dates = df_wide.index
    time_deltas_days = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]

    # Use last 30 days for velocity calculation (more stable)
    last_30_days = df_wide.index[-1] - timedelta(days=30)
    recent_data = df_wide[df_wide.index >= last_30_days]

    if len(recent_data) < 2:
        recent_data = df_wide.tail(5)  # Fallback to last 5 scans

    metrics = {}

    for metric_name in ['MuscleMass', 'FatMass', 'BodyWeight']:
        if metric_name not in df_wide.columns:
            continue

        # Linear regression on recent data
        y = recent_data[metric_name].values
        x = np.arange(len(y))

        if len(x) >= 2:
            # Fit linear trend
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]  # kg per scan

            # Convert to kg per week (average ~7 days between scans)
            avg_days_between = np.mean(time_deltas_days[-len(recent_data)+1:]) if time_deltas_days else 7
            velocity_per_week = slope * (7 / avg_days_between)

            # Calculate acceleration (change in velocity over last 2 months vs previous 2 months)
            if len(df_wide) >= 20:
                # Last 2 months
                recent_60d = df_wide.index[-1] - timedelta(days=60)
                data_60d = df_wide[df_wide.index >= recent_60d]

                # Previous 2 months
                previous_120d = df_wide.index[-1] - timedelta(days=120)
                previous_60d = df_wide.index[-1] - timedelta(days=60)
                data_prev_60d = df_wide[(df_wide.index >= previous_120d) & (df_wide.index < previous_60d)]

                if len(data_60d) >= 2 and len(data_prev_60d) >= 2:
                    # Velocity in recent 60d
                    y_recent = data_60d[metric_name].values
                    x_recent = np.arange(len(y_recent))
                    slope_recent = np.polyfit(x_recent, y_recent, 1)[0] * (7 / avg_days_between)

                    # Velocity in previous 60d
                    y_prev = data_prev_60d[metric_name].values
                    x_prev = np.arange(len(y_prev))
                    slope_prev = np.polyfit(x_prev, y_prev, 1)[0] * (7 / avg_days_between)

                    acceleration = slope_recent - slope_prev
                else:
                    acceleration = 0
            else:
                acceleration = 0

            metrics[metric_name] = {
                'velocity': velocity_per_week,
                'acceleration': acceleration,
                'trend': 'increasing' if velocity_per_week > 0 else 'decreasing' if velocity_per_week < 0 else 'stable'
            }

    return metrics


def calculate_trend_projections(df_wide, projection_days=90):
    """
    Calculate linear regression projections for muscle, fat, and weight
    Returns projected values for next N days with confidence intervals
    """
    if len(df_wide) < 5:
        return {}

    projections = {}

    for metric_name in ['MuscleMass', 'FatMass', 'BodyWeight']:
        if metric_name not in df_wide.columns:
            continue

        # Use last 90 days for projection (balance between recency and stability)
        last_90_days = df_wide.index[-1] - timedelta(days=90)
        recent_data = df_wide[df_wide.index >= last_90_days]

        if len(recent_data) < 3:
            recent_data = df_wide  # Use all data if <90 days available

        # Prepare data for regression
        y = recent_data[metric_name].values
        dates = recent_data.index
        x = np.array([(d - dates[0]).days for d in dates])

        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        # Calculate R² for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate standard error for confidence intervals
        residuals = y - y_pred
        std_error = np.std(residuals)

        # Generate projections
        last_date = df_wide.index[-1]
        projection_dates = [last_date + timedelta(days=i) for i in range(1, projection_days + 1, 7)]  # Weekly projections
        projection_x = np.array([(d - dates[0]).days for d in projection_dates])

        projected_values = slope * projection_x + intercept

        # 95% confidence interval (±1.96 * std_error)
        ci_upper = projected_values + (1.96 * std_error)
        ci_lower = projected_values - (1.96 * std_error)

        projections[metric_name] = {
            'dates': projection_dates,
            'values': projected_values.tolist(),
            'ci_upper': ci_upper.tolist(),
            'ci_lower': ci_lower.tolist(),
            'slope': slope,
            'r_squared': r_squared,
            'current_value': df_wide[metric_name].iloc[-1]
        }

    return projections


def estimate_time_to_goal(df_wide, goal_muscle=None, goal_fat=None, goal_weight=None):
    """
    Estimate time to reach goals based on current velocity
    Returns days to goal and target date
    """
    velocity = calculate_velocity_metrics(df_wide)
    estimates = {}

    current_values = {
        'MuscleMass': df_wide['MuscleMass'].iloc[-1] if 'MuscleMass' in df_wide.columns else None,
        'FatMass': df_wide['FatMass'].iloc[-1] if 'FatMass' in df_wide.columns else None,
        'BodyWeight': df_wide['BodyWeight'].iloc[-1] if 'BodyWeight' in df_wide.columns else None
    }

    goals = {
        'MuscleMass': goal_muscle,
        'FatMass': goal_fat,
        'BodyWeight': goal_weight
    }

    for metric_name, goal_value in goals.items():
        if goal_value is None or current_values[metric_name] is None:
            continue

        if metric_name not in velocity:
            continue

        vel = velocity[metric_name]['velocity']
        current = current_values[metric_name]
        difference = goal_value - current

        if abs(vel) < 0.01:  # Essentially no change
            estimates[metric_name] = {
                'achievable': False,
                'reason': 'No current progress trend',
                'recommendation': 'Adjust diet or training to create movement toward goal'
            }
        elif (difference > 0 and vel < 0) or (difference < 0 and vel > 0):
            estimates[metric_name] = {
                'achievable': False,
                'reason': f'Moving away from goal (velocity: {vel:+.2f} kg/week)',
                'recommendation': 'Reverse current trend to move toward goal'
            }
        else:
            weeks_to_goal = abs(difference / vel)
            days_to_goal = int(weeks_to_goal * 7)
            target_date = df_wide.index[-1] + timedelta(days=days_to_goal)

            # Check if realistic (not too far in future or past)
            if days_to_goal < 0 or days_to_goal > 730:  # Max 2 years
                estimates[metric_name] = {
                    'achievable': False,
                    'reason': f'Goal would take {days_to_goal} days' if days_to_goal > 730 else 'Goal already achieved',
                    'recommendation': 'Set a more realistic intermediate goal' if days_to_goal > 730 else 'Set new goal'
                }
            else:
                estimates[metric_name] = {
                    'achievable': True,
                    'days_to_goal': days_to_goal,
                    'weeks_to_goal': weeks_to_goal,
                    'target_date': target_date,
                    'current_value': current,
                    'goal_value': goal_value,
                    'velocity': vel,
                    'confidence': 'high' if abs(vel) > 0.1 else 'moderate' if abs(vel) > 0.05 else 'low'
                }

    return estimates


# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def create_kpi_card(title, value, subtitle, emoji="", highlight=False, wow_text="", plateau_warning=""):
    """Create a styled KPI card with optional week-over-week indicator and plateau warning"""
    card_class = "kpi-card"
    if highlight:
        card_class += " kpi-card--leader"

    wow_element = html.P(wow_text, style={
        'fontSize': '0.75rem',
        'color': '#5a6b8a',
        'margin': '4px 0 0 0',
        'fontWeight': '600'
    }) if wow_text else None

    # Plateau warning badge
    plateau_element = html.Div(plateau_warning, style={
        'fontSize': '0.7rem',
        'color': '#fff',
        'backgroundColor': '#ff6b6b',
        'padding': '4px 8px',
        'borderRadius': '4px',
        'marginTop': '8px',
        'fontWeight': '700',
        'textAlign': 'center'
    }) if plateau_warning else None

    return html.Div([
        html.H3(f"{emoji} {title}", className="kpi-card__title"),
        html.P(value, className="kpi-card__value"),
        html.P(subtitle, className="kpi-card__subtitle"),
        wow_element,
        plateau_element
    ], className=card_class)


def create_composition_trend_chart(df_wide, show_ma=True):
    """Create main composition trends chart with optional moving averages"""
    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['MuscleMass'],
        name='Muscle Mass',
        line=dict(color='#83AE00', width=2),
        mode='lines',
        opacity=0.4 if show_ma else 1,
        hovertemplate='<b>Muscle</b><br>%{y:.1f} kg<br>%{x|%b %d, %Y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['FatMass'],
        name='Fat Mass',
        line=dict(color='#852160', width=2),
        mode='lines',
        opacity=0.4 if show_ma else 1,
        hovertemplate='<b>Fat</b><br>%{y:.1f} kg<br>%{x|%b %d, %Y}<extra></extra>'
    ))

    # Bone Mass (baseline)
    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['BoneMass'],
        name='Bone Mass',
        line=dict(color='#dae0ee', width=2, dash='dash'),
        mode='lines',
        opacity=0.6,
        hovertemplate='<b>Bone</b><br>%{y:.1f} kg<extra></extra>'
    ))

    # 7-day moving averages
    if show_ma and 'MuscleMass_MA7' in df_wide.columns:
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['MuscleMass_MA7'],
            name='Muscle (7-day avg)',
            line=dict(color='#83AE00', width=3),
            mode='lines',
            hovertemplate='<b>Muscle (7-day avg)</b><br>%{y:.1f} kg<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['FatMass_MA7'],
            name='Fat (7-day avg)',
            line=dict(color='#852160', width=3),
            mode='lines',
            hovertemplate='<b>Fat (7-day avg)</b><br>%{y:.1f} kg<extra></extra>'
        ))

    fig.update_layout(
        title='Body Composition Over Time (with 7-Day Moving Average)',
        xaxis_title='Date',
        yaxis_title='Mass (kg)',
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )

    return fig


def create_recomposition_divergence_chart(df_wide):
    """Create dual-axis chart showing muscle up, fat down (from original app.py)"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['MuscleMass'],
        name='Muscle Mass',
        line=dict(color='#83AE00', width=3),
        mode='lines+markers',
        yaxis='y1',
        hovertemplate='<b>Muscle</b><br>%{y:.1f} kg<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['FatMass'],
        name='Fat Mass',
        line=dict(color='#852160', width=3),
        mode='lines+markers',
        yaxis='y2',
        hovertemplate='<b>Fat</b><br>%{y:.1f} kg<extra></extra>'
    ))

    fig.update_layout(
        title='Recomposition Progress: Muscle ↗ vs Fat ↘',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title=dict(text='Muscle Mass (kg)', font=dict(color='#83AE00')),
            tickfont=dict(color='#83AE00')
        ),
        yaxis2=dict(
            title=dict(text='Fat Mass (kg)', font=dict(color='#852160')),
            tickfont=dict(color='#852160'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )

    return fig


def create_segment_analysis_chart(df_wide, segment='Trunk', show_wow=True):
    """Create individual segment trend chart"""
    fig = go.Figure()

    muscle_col = f'{segment}MuscleMass'
    fat_col = f'{segment}FatMass'

    # Calculate changes
    muscle_change = ((df_wide[muscle_col].iloc[-1] - df_wide[muscle_col].iloc[0]) /
                     df_wide[muscle_col].iloc[0] * 100)
    fat_change = ((df_wide[fat_col].iloc[-1] - df_wide[fat_col].iloc[0]) /
                  df_wide[fat_col].iloc[0] * 100)

    # Muscle
    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide[muscle_col],
        name=f'{segment} Muscle',
        line=dict(color='#83AE00', width=2),
        fill='tonexty',
        hovertemplate='<b>Muscle</b><br>%{y:.1f} kg<extra></extra>'
    ))

    # Fat
    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide[fat_col],
        name=f'{segment} Fat',
        line=dict(color='#852160', width=2),
        fill='tozeroy',
        hovertemplate='<b>Fat</b><br>%{y:.1f} kg<extra></extra>'
    ))

    # Week-over-week
    title_text = f'{segment}: Muscle {muscle_change:+.1f}% | Fat {fat_change:+.1f}%'
    if show_wow and len(df_wide) > 1:
        latest_muscle = df_wide[muscle_col].iloc[-1]
        prev_muscle = df_wide[muscle_col].iloc[-2]
        wow_muscle = ((latest_muscle - prev_muscle) / prev_muscle * 100)
        title_text += f' | WoW: {wow_muscle:+.2f}%'

    fig.update_layout(
        title=title_text,
        xaxis_title='',
        yaxis_title='Mass (kg)',
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71', size=11),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        margin=dict(l=40, r=20, t=70, b=30),
        height=250
    )

    return fig


def create_health_indicators_chart(df_wide):
    """Create health metrics trend chart (from original app.py)"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['VisceralFatRating'],
        name='Visceral Fat',
        line=dict(color='#852160', width=2),
        mode='lines+markers',
        yaxis='y1',
        hovertemplate='<b>Visceral Fat</b><br>%{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['MetabolicAge'],
        name='Metabolic Age',
        line=dict(color='#83AE00', width=2),
        mode='lines+markers',
        yaxis='y2',
        hovertemplate='<b>Metabolic Age</b><br>%{y} years<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['Age'],
        name='Actual Age',
        line=dict(color='#dae0ee', width=1, dash='dash'),
        mode='lines',
        yaxis='y2',
        hovertemplate='<b>Actual Age</b><br>%{y} years<extra></extra>'
    ))

    fig.update_layout(
        title='Health Indicators Over Time',
        xaxis_title='Date',
        yaxis=dict(
            title=dict(text='Visceral Fat Rating', font=dict(color='#852160')),
            tickfont=dict(color='#852160')
        ),
        yaxis2=dict(
            title=dict(text='Age (years)', font=dict(color='#83AE00')),
            tickfont=dict(color='#83AE00'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=350
    )

    return fig


def create_phase_angle_chart(df_wide):
    """Create Phase Angle trend chart (from enhanced version)"""
    fig = go.Figure()

    if 'PhaseAngle_Avg' in df_wide.columns:
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['PhaseAngle_Avg'],
            name='Phase Angle',
            line=dict(color='#001c71', width=3),
            mode='lines+markers',
            marker=dict(size=6),
            hovertemplate='<b>Phase Angle</b><br>%{y:.1f}°<br>%{x|%b %d, %Y}<extra></extra>'
        ))

        fig.add_hline(y=5.0, line_dash="dash", line_color="#83AE00",
                     annotation_text="Good (>5.0°)", annotation_position="right")

    fig.update_layout(
        title='Phase Angle: Recovery & Training Readiness',
        xaxis_title='Date',
        yaxis_title='Phase Angle (degrees)',
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71'),
        margin=dict(l=50, r=50, t=80, b=50),
        height=350
    )

    return fig


def create_body_composition_ratios_chart(df_wide):
    """Create advanced body composition ratios chart with multiple subplots"""
    from plotly.subplots import make_subplots

    # Create subplot figure with 5 rows
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=(
            'Fat-Free Mass % (Higher = Better Body Comp)',
            'Muscle:Bone Ratio (Structural Health)',
            'BMR per kg Muscle (Metabolic Efficiency)',
            'Muscle Quality Index (Strength Potential)',
            'Hydration Ratio (Cellular Health)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
    )

    # Color scheme
    colors = ['#83AE00', '#001c71', '#852160', '#5a6b8a', '#00a8cc']

    # 1. Fat-Free Mass Percentage
    if 'FatFreeMass_Pct' in df_wide.columns:
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['FatFreeMass_Pct'],
            name='FFM %',
            line=dict(color=colors[0], width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='<b>Fat-Free Mass</b><br>%{y:.1f}%<extra></extra>',
            showlegend=False
        ), row=1, col=1)

    # 2. Muscle:Bone Ratio
    if 'MuscleBone_Ratio' in df_wide.columns:
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['MuscleBone_Ratio'],
            name='Muscle:Bone',
            line=dict(color=colors[1], width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='<b>Muscle:Bone Ratio</b><br>%{y:.2f}:1<extra></extra>',
            showlegend=False
        ), row=2, col=1)

    # 3. BMR per kg Muscle
    if 'BMR_per_kg_Muscle' in df_wide.columns:
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['BMR_per_kg_Muscle'],
            name='BMR/kg Muscle',
            line=dict(color=colors[2], width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='<b>BMR per kg Muscle</b><br>%{y:.0f} kJ/kg<extra></extra>',
            showlegend=False
        ), row=3, col=1)

    # 4. Muscle Quality Index
    if 'Muscle_Quality_Index' in df_wide.columns:
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['Muscle_Quality_Index'],
            name='MQI',
            line=dict(color=colors[3], width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='<b>Muscle Quality Index</b><br>%{y:.1f}<extra></extra>',
            showlegend=False
        ), row=4, col=1)

    # 5. Hydration Ratio
    if 'Hydration_Ratio' in df_wide.columns:
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['Hydration_Ratio'],
            name='Hydration',
            line=dict(color=colors[4], width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='<b>Hydration Ratio</b><br>%{y:.2f}<extra></extra>',
            showlegend=False
        ), row=5, col=1)

    # Update all y-axes
    fig.update_yaxes(title_text="FFM %", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="kJ/kg", row=3, col=1)
    fig.update_yaxes(title_text="MQI", row=4, col=1)
    fig.update_yaxes(title_text="Ratio", row=5, col=1)

    # Update layout
    fig.update_layout(
        title_text='Advanced Body Composition Ratios & Efficiency Metrics',
        title_font_size=16,
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71', size=10),
        height=1200,
        margin=dict(l=60, r=30, t=100, b=50),
        hovermode='x unified'
    )

    # Update all subplot titles to be smaller
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color='#001c71')

    return fig


def create_segmental_phase_angle_chart(df_wide):
    """Create chart showing Phase Angle by body segment (arms vs legs)"""
    fig = go.Figure()

    # Get phase angle columns
    phase_cols = {
        'LeftArm': 'PhaseAngleLeftArm',
        'RightArm': 'PhaseAngleRightArm',
        'LeftLeg': 'PhaseAngleLeftLeg',
        'RightLeg': 'PhaseAngleRightLeg'
    }

    colors_map = {
        'LeftArm': '#83AE00',
        'RightArm': '#6a8f00',
        'LeftLeg': '#001c71',
        'RightLeg': '#001549'
    }

    for segment, col in phase_cols.items():
        if col in df_wide.columns:
            fig.add_trace(go.Scatter(
                x=df_wide.index,
                y=df_wide[col].abs(),
                name=segment,
                line=dict(color=colors_map[segment], width=2),
                mode='lines+markers',
                marker=dict(size=4),
                hovertemplate=f'<b>{segment}</b><br>%{{y:.2f}}°<extra></extra>'
            ))

    # Add reference line
    fig.add_hline(y=5.0, line_dash="dash", line_color="#ff6b6b",
                  annotation_text="Good (>5.0°)", annotation_position="right")

    fig.update_layout(
        title='Phase Angle by Body Segment: Arms vs Legs Recovery',
        xaxis_title='Date',
        yaxis_title='Phase Angle (degrees)',
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=350
    )

    return fig


def create_water_distribution_chart(df_wide):
    """Create chart showing intra vs extracellular water distribution"""
    fig = go.Figure()

    if 'IntraCellularWaterMass' in df_wide.columns and 'ExtraCellularWaterMass' in df_wide.columns:
        # Stacked area chart
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['IntraCellularWaterMass'],
            name='Intracellular Water',
            fill='tozeroy',
            line=dict(color='#001c71', width=2),
            hovertemplate='<b>ICW</b><br>%{y:.1f} kg<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['ExtraCellularWaterMass'],
            name='Extracellular Water',
            fill='tonexty',
            line=dict(color='#852160', width=2),
            hovertemplate='<b>ECW</b><br>%{y:.1f} kg<extra></extra>'
        ))

        # Add hydration ratio as line on secondary axis
        if 'Hydration_Ratio' in df_wide.columns:
            fig.add_trace(go.Scatter(
                x=df_wide.index,
                y=df_wide['Hydration_Ratio'],
                name='ICW:ECW Ratio',
                line=dict(color='#83AE00', width=3, dash='dash'),
                mode='lines+markers',
                marker=dict(size=6),
                yaxis='y2',
                hovertemplate='<b>Ratio</b><br>%{y:.2f}<extra></extra>'
            ))

            # Optimal ratio zones
            fig.add_hrect(y0=1.3, y1=1.6,
                         annotation_text="Optimal", annotation_position="right",
                         fillcolor="#83AE00", opacity=0.1, layer="below",
                         yref='y2', line_width=0)

    fig.update_layout(
        title='Water Distribution & Hydration Status',
        xaxis_title='Date',
        yaxis=dict(
            title=dict(text='Water Mass (kg)', font=dict(color='#001c71')),
            tickfont=dict(color='#001c71')
        ),
        yaxis2=dict(
            title=dict(text='Hydration Ratio (ICW:ECW)', font=dict(color='#83AE00')),
            tickfont=dict(color='#83AE00'),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )

    return fig


def create_recovery_score_chart(df_wide):
    """Create recovery score and training readiness chart"""
    fig = go.Figure()

    if 'Recovery_Score' in df_wide.columns:
        # Recovery score line
        fig.add_trace(go.Scatter(
            x=df_wide.index,
            y=df_wide['Recovery_Score'],
            name='Recovery Score',
            line=dict(color='#001c71', width=3),
            mode='lines+markers',
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(0, 28, 113, 0.1)',
            hovertemplate='<b>Recovery Score</b><br>%{y:.0f}/100<extra></extra>'
        ))

        # Reference zones
        fig.add_hrect(y0=80, y1=100, fillcolor="#28a745", opacity=0.1, layer="below",
                     annotation_text="READY", annotation_position="right", line_width=0)
        fig.add_hrect(y0=70, y1=80, fillcolor="#6c757d", opacity=0.1, layer="below",
                     annotation_text="GOOD", annotation_position="right", line_width=0)
        fig.add_hrect(y0=60, y1=70, fillcolor="#ffc107", opacity=0.1, layer="below",
                     annotation_text="MODERATE", annotation_position="right", line_width=0)
        fig.add_hrect(y0=50, y1=60, fillcolor="#fd7e14", opacity=0.1, layer="below",
                     annotation_text="CAUTION", annotation_position="right", line_width=0)
        fig.add_hrect(y0=0, y1=50, fillcolor="#dc3545", opacity=0.1, layer="below",
                     annotation_text="REST", annotation_position="right", line_width=0)

    fig.update_layout(
        title='Recovery Score & Training Readiness (0-100)',
        xaxis_title='Date',
        yaxis_title='Recovery Score',
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )

    return fig


def create_projection_chart(df_wide, projections, metric_name='MuscleMass'):
    """Create chart showing historical data with future projections and confidence intervals"""
    fig = go.Figure()

    if metric_name not in df_wide.columns or metric_name not in projections:
        return fig

    proj = projections[metric_name]

    # Historical data
    color_map = {'MuscleMass': '#83AE00', 'FatMass': '#852160', 'BodyWeight': '#001c71'}
    color = color_map.get(metric_name, '#001c71')

    fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide[metric_name],
        name='Historical Data',
        line=dict(color=color, width=3),
        mode='lines+markers',
        marker=dict(size=5),
        hovertemplate='<b>Actual</b><br>%{y:.1f} kg<br>%{x|%b %d, %Y}<extra></extra>'
    ))

    # Projected trend line
    fig.add_trace(go.Scatter(
        x=proj['dates'],
        y=proj['values'],
        name='Projected Trend',
        line=dict(color=color, width=3, dash='dash'),
        mode='lines',
        hovertemplate='<b>Projection</b><br>%{y:.1f} kg<br>%{x|%b %d, %Y}<extra></extra>'
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=proj['dates'] + proj['dates'][::-1],
        y=proj['ci_upper'] + proj['ci_lower'][::-1],
        fill='toself',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='95% Confidence Interval',
        showlegend=True
    ))

    # Add current value marker
    fig.add_trace(go.Scatter(
        x=[df_wide.index[-1]],
        y=[proj['current_value']],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name='Current',
        hovertemplate='<b>Current Value</b><br>%{y:.1f} kg<extra></extra>'
    ))

    metric_labels = {'MuscleMass': 'Muscle Mass', 'FatMass': 'Fat Mass', 'BodyWeight': 'Body Weight'}
    label = metric_labels.get(metric_name, metric_name)

    fig.update_layout(
        title=f'{label} Projection (90-day forecast, R²={proj["r_squared"]:.3f})',
        xaxis_title='Date',
        yaxis_title='Mass (kg)',
        hovermode='x unified',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        height=400
    )

    return fig


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

# Load config and get last used file
config = load_config()
current_filepath = config.get('last_file', 'data/BoditraxAccount_20251006_102227.csv')

df_long = load_boditrax_data(current_filepath)
df_wide_full = transform_to_wide_format(df_long)
df_wide_full = calculate_phase_angle_average(df_wide_full)
df_wide_full = calculate_body_composition_ratios(df_wide_full)
df_wide_full = calculate_recovery_score(df_wide_full)
df_wide_full = add_moving_averages(df_wide_full, window=7)

# ============================================================================
# DASH APP
# ============================================================================

app = dash.Dash(
    __name__,
    title="Body Recomposition Tracker - Complete",
    update_title=None,
    external_stylesheets=['assets/jazz_theme.css']
)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Body Recomposition Tracker", className="page-title"),
        html.P("Complete version with all features: moving averages, phase angle, filtering, and comprehensive analysis",
               className="page-subtitle")
    ]),

    # File Management Controls
    html.Div([
        html.Div([
            html.Label("Load New Dataset:", className="filter-label"),
            dcc.Upload(
                id='upload-data',
                children=html.Button('📂 Select Boditrax CSV File', className='upload-button'),
                multiple=False
            ),
            html.Div(id='upload-status', style={'marginTop': '10px', 'color': '#5a6b8a', 'fontSize': '14px'})
        ], className="filter-group"),

        html.Div([
            html.Label("Export Report:", className="filter-label"),
            html.Button('📄 Export to PDF', id='export-pdf-button', className='export-button'),
            dcc.Download(id='download-pdf')
        ], className="filter-group"),
    ], className="filters", style={'marginBottom': '20px'}),

    # Date Range Filter
    html.Div([
        html.Div([
            html.Label("Date Range:", className="filter-label"),
            dcc.Dropdown(
                id='date-range-preset',
                options=[
                    {'label': 'All Time', 'value': 'all'},
                    {'label': 'Last 30 Days', 'value': '30'},
                    {'label': 'Last 90 Days', 'value': '90'},
                    {'label': 'Last 6 Months', 'value': '180'},
                    {'label': 'Last Year', 'value': '365'},
                    {'label': 'Custom Range', 'value': 'custom'}
                ],
                value='all',
                className='jazz-dropdown',
                clearable=False
            )
        ], className="filter-group"),

        html.Div([
            html.Label("Custom Start Date:", className="filter-label"),
            dcc.DatePickerSingle(
                id='custom-start-date',
                date=df_wide_full.index[0].date(),
                min_date_allowed=df_wide_full.index[0].date(),
                max_date_allowed=df_wide_full.index[-1].date(),
                disabled=True
            )
        ], className="filter-group"),

        html.Div([
            html.Label("Custom End Date:", className="filter-label"),
            dcc.DatePickerSingle(
                id='custom-end-date',
                date=df_wide_full.index[-1].date(),
                min_date_allowed=df_wide_full.index[0].date(),
                max_date_allowed=df_wide_full.index[-1].date(),
                disabled=True
            )
        ], className="filter-group"),
    ], className="filters"),

    # KPI Cards
    html.Div(id='kpi-cards', className="kpi-wrapper"),

    # Plateau Alerts Section (NEW)
    html.Div(id='plateau-alerts', style={'marginTop': '20px', 'marginBottom': '20px'}),

    # Main Composition Trends (with 7-day MA)
    html.Div([
        html.Div([
            dcc.Graph(id='composition-trend')
        ], className="chart-card chart-card--full")
    ]),

    # Recomposition Divergence (from original)
    html.Div([
        html.Div([
            dcc.Graph(id='recomp-divergence')
        ], className="chart-card chart-card--full")
    ]),

    # Segment Analysis Section (Original 3)
    html.H2("Segment Analysis: Strengths & Weaknesses", className="section-title"),
    html.Div([
        html.Div([dcc.Graph(id='trunk-analysis')], className="chart-card"),
        html.Div([dcc.Graph(id='leftleg-analysis')], className="chart-card"),
        html.Div([dcc.Graph(id='leftarm-analysis')], className="chart-card"),
    ], className="chart-row"),

    # Additional Segment Analysis (Enhanced - Right side comparisons)
    html.H2("Symmetry Analysis: Left vs Right Comparison", className="section-title"),
    html.Div([
        html.Div([dcc.Graph(id='rightleg-analysis')], className="chart-card"),
        html.Div([dcc.Graph(id='rightarm-analysis')], className="chart-card"),
        html.Div([dcc.Graph(id='combined-legs-analysis')], className="chart-card"),
    ], className="chart-row"),

    # Health Indicators (from original)
    html.H2("Health Indicators", className="section-title"),
    html.Div([
        html.Div([
            dcc.Graph(id='health-indicators')
        ], className="chart-card chart-card--full")
    ]),

    # Phase Angle (from enhanced)
    html.H2("Phase Angle: Recovery & Training Readiness", className="section-title"),
    html.Div([
        html.Div([
            dcc.Graph(id='phase-angle-chart')
        ], className="chart-card chart-card--full")
    ]),

    # Advanced Body Composition Ratios
    html.H2("Advanced Body Composition Ratios", className="section-title"),
    html.P("Professional-grade metrics for deeper body composition analysis and efficiency tracking",
           style={'color': '#5a6b8a', 'fontSize': '0.95rem', 'marginTop': '-10px', 'marginBottom': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='body-comp-ratios-chart')
        ], className="chart-card chart-card--full")
    ]),

    # Recovery & Training Readiness Dashboard (NEW)
    html.H2("Recovery & Training Readiness", className="section-title"),
    html.P("Optimize training timing with Phase Angle, hydration analysis, and recovery metrics",
           style={'color': '#5a6b8a', 'fontSize': '0.95rem', 'marginTop': '-10px', 'marginBottom': '20px'}),

    # Training Readiness Indicator
    html.Div(id='training-readiness-indicator', style={'marginBottom': '20px'}),

    # Recovery Score Chart
    html.Div([
        html.Div([
            dcc.Graph(id='recovery-score-chart')
        ], className="chart-card chart-card--full")
    ]),

    # Segmental Phase Angle Chart
    html.Div([
        html.Div([
            dcc.Graph(id='segmental-phase-angle-chart')
        ], className="chart-card chart-card--full")
    ]),

    # Water Distribution Chart
    html.Div([
        html.Div([
            dcc.Graph(id='water-distribution-chart')
        ], className="chart-card chart-card--full")
    ]),

    # Predictive Analytics & Goal Tracking (NEW)
    html.H2("Predictive Analytics & Goal Tracking", className="section-title"),
    html.P("Trend projections, velocity metrics, and goal achievement estimates",
           style={'color': '#5a6b8a', 'fontSize': '0.95rem', 'marginTop': '-10px', 'marginBottom': '20px'}),

    # Velocity/Acceleration Metrics
    html.Div(id='velocity-metrics-display', style={'marginBottom': '20px'}),

    # Projection Charts
    html.Div([
        html.Div([
            dcc.Graph(id='muscle-projection-chart')
        ], className="chart-card chart-card--full")
    ]),

    html.Div([
        html.Div([
            dcc.Graph(id='fat-projection-chart')
        ], className="chart-card chart-card--full")
    ]),

    html.Div([
        html.Div([
            dcc.Graph(id='weight-projection-chart')
        ], className="chart-card chart-card--full")
    ]),

    # Goal Tracker (Static example - could be made interactive)
    html.Div(id='goal-tracker-display', style={'marginTop': '20px', 'marginBottom': '20px'}),

    # Data Table
    html.H2("Scan History Details", className="section-title"),
    html.Div([
        dash_table.DataTable(
            id='scan-table',
            page_size=10,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '12px',
                'fontFamily': 'Neue Haas, Helvetica Neue, Arial, sans-serif',
                'fontSize': '14px'
            },
            style_header={
                'backgroundColor': '#001c71',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9fbfc'}
            ]
        )
    ], className="table-card jazz-table"),

], className="jazz-app")


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('custom-start-date', 'disabled'),
     Output('custom-end-date', 'disabled')],
    [Input('date-range-preset', 'value')]
)
def toggle_custom_dates(preset):
    """Enable/disable custom date pickers"""
    return preset != 'custom', preset != 'custom'


@app.callback(
    [Output('kpi-cards', 'children'),
     Output('plateau-alerts', 'children'),
     Output('composition-trend', 'figure'),
     Output('recomp-divergence', 'figure'),
     Output('trunk-analysis', 'figure'),
     Output('leftleg-analysis', 'figure'),
     Output('leftarm-analysis', 'figure'),
     Output('rightleg-analysis', 'figure'),
     Output('rightarm-analysis', 'figure'),
     Output('combined-legs-analysis', 'figure'),
     Output('health-indicators', 'figure'),
     Output('phase-angle-chart', 'figure'),
     Output('body-comp-ratios-chart', 'figure'),
     Output('training-readiness-indicator', 'children'),
     Output('recovery-score-chart', 'figure'),
     Output('segmental-phase-angle-chart', 'figure'),
     Output('water-distribution-chart', 'figure'),
     Output('velocity-metrics-display', 'children'),
     Output('muscle-projection-chart', 'figure'),
     Output('fat-projection-chart', 'figure'),
     Output('weight-projection-chart', 'figure'),
     Output('goal-tracker-display', 'children'),
     Output('scan-table', 'columns'),
     Output('scan-table', 'data')],
    [Input('date-range-preset', 'value'),
     Input('custom-start-date', 'date'),
     Input('custom-end-date', 'date')]
)
def update_dashboard(preset, custom_start, custom_end):
    """Update all dashboard components based on date filter"""

    # Filter data based on selection
    df_wide = df_wide_full.copy()

    if preset == 'custom' and custom_start and custom_end:
        start_date = pd.to_datetime(custom_start)
        end_date = pd.to_datetime(custom_end)
        df_wide = df_wide[(df_wide.index >= start_date) & (df_wide.index <= end_date)]
    elif preset != 'all':
        days = int(preset)
        end_date = df_wide.index[-1]
        start_date = end_date - timedelta(days=days)
        df_wide = df_wide[df_wide.index >= start_date]

    # Calculate metrics
    metrics = calculate_recomp_metrics(df_wide)

    # Detect plateaus
    plateaus = detect_plateaus(df_wide)

    # Determine plateau warnings for KPI cards
    muscle_warning = ""
    fat_warning = ""
    if plateaus['muscle_30d']:
        muscle_warning = "⚠️ STALLED 30d"
    elif plateaus['muscle_21d']:
        muscle_warning = "⚠️ PLATEAU 21d"
    elif plateaus['muscle_14d']:
        muscle_warning = "⚠️ SLOWING 14d"

    if plateaus['fat_30d']:
        fat_warning = "⚠️ STALLED 30d"
    elif plateaus['fat_21d']:
        fat_warning = "⚠️ PLATEAU 21d"
    elif plateaus['fat_14d']:
        fat_warning = "⚠️ SLOWING 14d"

    # KPI Cards (Original 4 from app.py with WoW from enhanced + plateau warnings)
    kpi_cards = [
        create_kpi_card(
            "MUSCLE GAIN",
            f"{metrics['muscle_change']:+.1f} kg",
            f"({metrics['muscle_pct_change']:+.1f}%)",
            "💪",
            highlight=metrics['muscle_change'] > 0,
            wow_text=f"WoW: {metrics['muscle_wow']:+.2f} kg ({metrics['muscle_wow_pct']:+.1f}%)",
            plateau_warning=muscle_warning
        ),
        create_kpi_card(
            "FAT LOSS",
            f"{metrics['fat_change']:+.1f} kg",
            f"({metrics['fat_pct_change']:+.1f}%)",
            "🔥",
            highlight=metrics['fat_change'] < 0,
            wow_text=f"WoW: {metrics['fat_wow']:+.2f} kg ({metrics['fat_wow_pct']:+.1f}%)",
            plateau_warning=fat_warning
        ),
        create_kpi_card(
            "NET WEIGHT",
            f"{metrics['weight_change']:+.1f} kg",
            f"({metrics['weight_pct_change']:+.1f}%)",
            "⚖️",
            wow_text=f"WoW: {metrics['weight_wow']:+.2f} kg"
        ),
        create_kpi_card(
            "RECOMP RATIO",
            f"{metrics['recomp_ratio']:.2f}:1" if metrics['recomp_ratio'] > 0 else "N/A",
            metrics['recomp_status'],
            "🎯",
            highlight=metrics['recomp_ratio'] > 0.8
        ),
    ]

    # Plateau Alerts Section
    if plateaus['alerts']:
        severity_colors = {
            'HIGH': '#dc3545',
            'MEDIUM': '#fd7e14',
            'LOW': '#ffc107',
            'INFO': '#28a745'
        }

        alert_cards = []
        for alert in plateaus['alerts']:
            severity = alert['severity']
            bg_color = severity_colors.get(severity, '#6c757d')

            alert_cards.append(
                html.Div([
                    html.Div([
                        html.H4(alert['message'], style={
                            'margin': '0 0 8px 0',
                            'fontSize': '1rem',
                            'fontWeight': '700'
                        }),
                        html.P(f"📊 {alert['metric']}", style={
                            'margin': '0 0 8px 0',
                            'fontSize': '0.85rem',
                            'opacity': '0.9'
                        }),
                        html.P(f"💡 {alert['recommendation']}", style={
                            'margin': '0',
                            'fontSize': '0.85rem',
                            'fontStyle': 'italic'
                        })
                    ], style={
                        'backgroundColor': bg_color,
                        'color': '#fff',
                        'padding': '16px 20px',
                        'borderRadius': '8px',
                        'marginBottom': '12px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ])
            )

        plateau_section = html.Div([
            html.H3("🚨 Progress Alerts", style={
                'fontSize': '1.25rem',
                'marginBottom': '12px',
                'color': '#001c71',
                'fontWeight': '700'
            }),
            html.Div(alert_cards)
        ])
    else:
        # No plateaus detected - show success message
        plateau_section = html.Div([
            html.Div([
                html.H4("✅ No Plateaus Detected", style={
                    'margin': '0 0 4px 0',
                    'fontSize': '1rem',
                    'fontWeight': '700'
                }),
                html.P("All metrics showing healthy change. Keep up the great work!", style={
                    'margin': '0',
                    'fontSize': '0.85rem'
                })
            ], style={
                'backgroundColor': '#28a745',
                'color': '#fff',
                'padding': '16px 20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ])

    # Charts
    comp_fig = create_composition_trend_chart(df_wide, show_ma=True)
    recomp_fig = create_recomposition_divergence_chart(df_wide)
    trunk_fig = create_segment_analysis_chart(df_wide, 'Trunk', show_wow=True)
    leftleg_fig = create_segment_analysis_chart(df_wide, 'LeftLeg', show_wow=True)
    leftarm_fig = create_segment_analysis_chart(df_wide, 'LeftArm', show_wow=True)
    rightleg_fig = create_segment_analysis_chart(df_wide, 'RightLeg', show_wow=True)
    rightarm_fig = create_segment_analysis_chart(df_wide, 'RightArm', show_wow=True)

    # Combined legs
    legs_fig = go.Figure()
    legs_fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['LeftLegMuscleMass'] + df_wide['RightLegMuscleMass'],
        name='Total Leg Muscle',
        line=dict(color='#83AE00', width=3),
        hovertemplate='%{y:.1f} kg<extra></extra>'
    ))
    legs_fig.add_trace(go.Scatter(
        x=df_wide.index,
        y=df_wide['LeftLegFatMass'] + df_wide['RightLegFatMass'],
        name='Total Leg Fat',
        line=dict(color='#852160', width=3),
        hovertemplate='%{y:.1f} kg<extra></extra>'
    ))
    legs_fig.update_layout(
        title='Combined Legs: Total Muscle & Fat',
        xaxis_title='',
        yaxis_title='Mass (kg)',
        plot_bgcolor='#f9fbfc',
        paper_bgcolor='#ffffff',
        font=dict(family="Neue Haas, Helvetica Neue, Arial, sans-serif", color='#001c71', size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        margin=dict(l=40, r=20, t=70, b=30),
        height=250
    )

    health_fig = create_health_indicators_chart(df_wide)
    phase_fig = create_phase_angle_chart(df_wide)
    body_comp_ratios_fig = create_body_composition_ratios_chart(df_wide)

    # Training Readiness Indicator
    latest_recovery_score = df_wide['Recovery_Score'].iloc[-1] if 'Recovery_Score' in df_wide.columns else 75
    latest_readiness_status = df_wide['Readiness_Status'].iloc[-1] if 'Readiness_Status' in df_wide.columns else 'GOOD'

    readiness_colors = {
        'READY': '#28a745',
        'GOOD': '#6c757d',
        'MODERATE': '#ffc107',
        'CAUTION': '#fd7e14',
        'REST NEEDED': '#dc3545'
    }

    readiness_icons = {
        'READY': '🟢',
        'GOOD': '🟢',
        'MODERATE': '🟡',
        'CAUTION': '🟠',
        'REST NEEDED': '🔴'
    }

    readiness_messages = {
        'READY': 'Perfect time for high-intensity training',
        'GOOD': 'Ready for normal training volume',
        'MODERATE': 'Consider lighter training or active recovery',
        'CAUTION': 'Limit training intensity - focus on technique',
        'REST NEEDED': 'Take rest day or very light activity only'
    }

    readiness_color = readiness_colors.get(latest_readiness_status, '#6c757d')
    readiness_icon = readiness_icons.get(latest_readiness_status, '⚪')
    readiness_message = readiness_messages.get(latest_readiness_status, 'Recovery data processing')

    training_readiness_indicator = html.Div([
        html.Div([
            html.Div([
                html.H3(f"{readiness_icon} {latest_readiness_status}", style={
                    'margin': '0 0 8px 0',
                    'fontSize': '1.5rem',
                    'fontWeight': '700'
                }),
                html.P(f"Recovery Score: {latest_recovery_score:.0f}/100", style={
                    'margin': '0 0 8px 0',
                    'fontSize': '1.1rem',
                    'fontWeight': '600'
                }),
                html.P(f"💡 {readiness_message}", style={
                    'margin': '0',
                    'fontSize': '0.95rem',
                    'fontStyle': 'italic',
                    'opacity': '0.9'
                })
            ], style={
                'backgroundColor': readiness_color,
                'color': '#fff',
                'padding': '24px 32px',
                'borderRadius': '12px',
                'textAlign': 'center',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.15)',
                'maxWidth': '600px',
                'margin': '0 auto'
            })
        ])
    ])

    # Recovery Charts
    recovery_score_fig = create_recovery_score_chart(df_wide)
    segmental_phase_angle_fig = create_segmental_phase_angle_chart(df_wide)
    water_distribution_fig = create_water_distribution_chart(df_wide)

    # Predictive Analytics
    velocity_metrics = calculate_velocity_metrics(df_wide)
    projections = calculate_trend_projections(df_wide, projection_days=90)

    # Velocity Metrics Display
    velocity_cards = []
    for metric_name, vel_data in velocity_metrics.items():
        metric_labels = {'MuscleMass': 'Muscle', 'FatMass': 'Fat', 'BodyWeight': 'Weight'}
        label = metric_labels.get(metric_name, metric_name)

        trend_icon = '📈' if vel_data['trend'] == 'increasing' else '📉' if vel_data['trend'] == 'decreasing' else '➡️'
        trend_color = '#28a745' if (metric_name == 'MuscleMass' and vel_data['trend'] == 'increasing') or (metric_name == 'FatMass' and vel_data['trend'] == 'decreasing') else '#dc3545' if vel_data['trend'] != 'stable' else '#6c757d'

        velocity_cards.append(
            html.Div([
                html.H4(f"{trend_icon} {label}", style={'margin': '0 0 8px 0', 'fontSize': '1.1rem'}),
                html.P(f"Velocity: {vel_data['velocity']:+.3f} kg/week", style={'margin': '0 0 4px 0', 'fontSize': '0.95rem', 'fontWeight': '600'}),
                html.P(f"Acceleration: {vel_data['acceleration']:+.4f} kg/week²", style={'margin': '0', 'fontSize': '0.85rem', 'opacity': '0.8'}),
            ], style={
                'backgroundColor': '#fff',
                'border': f'3px solid {trend_color}',
                'padding': '16px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'flex': '1',
                'minWidth': '200px'
            })
        )

    velocity_display = html.Div([
        html.H3("Current Velocity & Acceleration", style={'fontSize': '1.2rem', 'marginBottom': '12px', 'color': '#001c71'}),
        html.Div(velocity_cards, style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'})
    ])

    # Projection Charts
    muscle_projection_fig = create_projection_chart(df_wide, projections, 'MuscleMass')
    fat_projection_fig = create_projection_chart(df_wide, projections, 'FatMass')
    weight_projection_fig = create_projection_chart(df_wide, projections, 'BodyWeight')

    # Goal Tracker (Example goals - could be made interactive)
    # Using reasonable goals based on current progress
    current_muscle = df_wide['MuscleMass'].iloc[-1] if 'MuscleMass' in df_wide.columns else None
    current_fat = df_wide['FatMass'].iloc[-1] if 'FatMass' in df_wide.columns else None

    # Example: Gain 2kg muscle, lose 2kg fat
    goal_estimates = estimate_time_to_goal(
        df_wide,
        goal_muscle=current_muscle + 2 if current_muscle else None,
        goal_fat=current_fat - 2 if current_fat else None
    )

    goal_cards = []
    for metric_name, estimate in goal_estimates.items():
        metric_labels = {'MuscleMass': '💪 Muscle Goal (+2kg)', 'FatMass': '🔥 Fat Goal (-2kg)', 'BodyWeight': '⚖️ Weight Goal'}
        label = metric_labels.get(metric_name, metric_name)

        if estimate['achievable']:
            goal_cards.append(
                html.Div([
                    html.H4(label, style={'margin': '0 0 8px 0', 'fontSize': '1.1rem'}),
                    html.P(f"Target: {estimate['goal_value']:.1f} kg", style={'margin': '0 0 4px 0', 'fontSize': '0.9rem'}),
                    html.P(f"Est. Time: {estimate['weeks_to_goal']:.1f} weeks ({estimate['days_to_goal']} days)", style={'margin': '0 0 4px 0', 'fontSize': '0.9rem', 'fontWeight': '600'}),
                    html.P(f"Target Date: {estimate['target_date'].strftime('%b %d, %Y')}", style={'margin': '0 0 4px 0', 'fontSize': '0.85rem'}),
                    html.P(f"Confidence: {estimate['confidence'].upper()}", style={'margin': '0', 'fontSize': '0.8rem', 'opacity': '0.7'}),
                ], style={
                    'backgroundColor': '#d4edda',
                    'border': '2px solid #28a745',
                    'padding': '16px',
                    'borderRadius': '8px',
                    'flex': '1',
                    'minWidth': '250px'
                })
            )
        else:
            goal_cards.append(
                html.Div([
                    html.H4(label, style={'margin': '0 0 8px 0', 'fontSize': '1.1rem'}),
                    html.P(f"⚠️ {estimate['reason']}", style={'margin': '0 0 8px 0', 'fontSize': '0.9rem', 'fontWeight': '600', 'color': '#dc3545'}),
                    html.P(f"💡 {estimate['recommendation']}", style={'margin': '0', 'fontSize': '0.85rem', 'fontStyle': 'italic'}),
                ], style={
                    'backgroundColor': '#f8d7da',
                    'border': '2px solid #dc3545',
                    'padding': '16px',
                    'borderRadius': '8px',
                    'flex': '1',
                    'minWidth': '250px'
                })
            )

    goal_tracker = html.Div([
        html.H3("Goal Achievement Estimates", style={'fontSize': '1.2rem', 'marginBottom': '12px', 'color': '#001c71'}),
        html.P("Based on current velocity (example goals: +2kg muscle, -2kg fat)", style={'fontSize': '0.85rem', 'color': '#5a6b8a', 'marginBottom': '12px'}),
        html.Div(goal_cards, style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'})
    ]) if goal_cards else html.Div()

    # Table
    table_columns = [
        {'name': 'Date', 'id': 'Date'},
        {'name': 'Weight', 'id': 'BodyWeight'},
        {'name': 'Muscle', 'id': 'MuscleMass'},
        {'name': 'Fat', 'id': 'FatMass'},
        {'name': 'BMI', 'id': 'BodyMassIndex'},
        {'name': 'Visc. Fat', 'id': 'VisceralFatRating'},
        {'name': 'Meta Age', 'id': 'MetabolicAge'},
        {'name': 'BMR (kJ)', 'id': 'BasalMetabolicRatekJ'},
    ]

    table_data = df_wide.reset_index().round(1).to_dict('records')

    return (kpi_cards, plateau_section, comp_fig, recomp_fig, trunk_fig, leftleg_fig, leftarm_fig,
            rightleg_fig, rightarm_fig, legs_fig, health_fig, phase_fig,
            body_comp_ratios_fig, training_readiness_indicator, recovery_score_fig,
            segmental_phase_angle_fig, water_distribution_fig, velocity_display,
            muscle_projection_fig, fat_projection_fig, weight_projection_fig, goal_tracker, table_columns, table_data)


# ============================================================================
# FILE UPLOAD CALLBACK
# ============================================================================

@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_file_upload(contents, filename):
    """Handle CSV file upload and save to data folder"""
    if contents is None:
        return ""

    try:
        # Decode the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Save to data folder with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"BoditraxAccount_{timestamp}.csv"
        filepath = os.path.join('data', new_filename)

        with open(filepath, 'wb') as f:
            f.write(decoded)

        # Update config with new file
        config = load_config()
        config['last_file'] = filepath
        save_config(config)

        return html.Div([
            html.Span("✅ ", style={'color': '#83AE00', 'fontWeight': 'bold'}),
            html.Span(f"File '{filename}' uploaded successfully as '{new_filename}'. "),
            html.Span("Please refresh the page to load the new data.", style={'fontWeight': 'bold', 'color': '#001c71'})
        ])

    except Exception as e:
        return html.Div([
            html.Span("❌ ", style={'color': '#852160', 'fontWeight': 'bold'}),
            html.Span(f"Error uploading file: {str(e)}", style={'color': '#852160'})
        ])


# ============================================================================
# PDF EXPORT CALLBACK
# ============================================================================

@app.callback(
    Output('download-pdf', 'data'),
    Input('export-pdf-button', 'n_clicks'),
    prevent_initial_call=True
)
def export_to_pdf(n_clicks):
    """Generate and download HTML report (can be printed to PDF)"""
    if n_clicks is None:
        return None

    # Calculate current metrics
    metrics = calculate_recomp_metrics(df_wide_full)
    latest_date = df_wide_full.index[-1].strftime('%Y-%m-%d')
    first_date = df_wide_full.index[0].strftime('%Y-%m-%d')

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Body Recomposition Report</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; margin: 40px; color: #001c71; }}
            h1 {{ color: #001c71; border-bottom: 3px solid #852160; padding-bottom: 10px; }}
            h2 {{ color: #852160; margin-top: 30px; }}
            .metric {{ margin: 15px 0; padding: 15px; background: #f9fbfc; border-left: 4px solid #83AE00; }}
            .metric-name {{ font-weight: bold; color: #001c71; }}
            .metric-value {{ font-size: 24px; color: #83AE00; font-weight: bold; }}
            .negative {{ color: #852160; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #001c71; color: white; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #852160; font-size: 12px; color: #5a6b8a; }}
        </style>
    </head>
    <body>
        <h1>Body Recomposition Progress Report</h1>
        <p><strong>Report Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>Data Period:</strong> {first_date} to {latest_date}</p>
        <p><strong>Total Scans:</strong> {len(df_wide_full)}</p>

        <h2>Key Achievements</h2>

        <div class="metric">
            <div class="metric-name">💪 Muscle Mass Change</div>
            <div class="metric-value">{metrics['muscle_change']:+.1f} kg ({metrics['muscle_change_pct']:+.1f}%)</div>
        </div>

        <div class="metric">
            <div class="metric-name">🔥 Fat Mass Change</div>
            <div class="metric-value negative">{metrics['fat_change']:+.1f} kg ({metrics['fat_change_pct']:+.1f}%)</div>
        </div>

        <div class="metric">
            <div class="metric-name">⚖️ Body Weight Change</div>
            <div class="metric-value">{metrics['weight_change']:+.1f} kg ({metrics['weight_change_pct']:+.1f}%)</div>
        </div>

        <div class="metric">
            <div class="metric-name">🎯 Recomposition Ratio</div>
            <div class="metric-value">{metrics['recomp_ratio']:.2f}:1</div>
            <p style="margin-top: 10px; font-size: 14px;">
                {
                    "Elite performance! Nearly 1:1 muscle gain to fat loss." if metrics['recomp_ratio'] >= 0.9 else
                    "Excellent! Strong recomposition progress." if metrics['recomp_ratio'] >= 0.7 else
                    "Good progress on body recomposition." if metrics['recomp_ratio'] >= 0.5 else
                    "Moderate recomposition progress."
                }
            </p>
        </div>

        <h2>Current Health Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Current Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Visceral Fat Rating</td>
                <td>{df_wide_full['VisceralFatRating'].iloc[-1]:.0f}</td>
                <td>{"✅ Excellent (≤10)" if df_wide_full['VisceralFatRating'].iloc[-1] <= 10 else "⚠️ Monitor"}</td>
            </tr>
            <tr>
                <td>Metabolic Age</td>
                <td>{df_wide_full['MetabolicAge'].iloc[-1]:.0f} years</td>
                <td>{"✅ Great" if df_wide_full['MetabolicAge'].iloc[-1] < 40 else "Good"}</td>
            </tr>
            <tr>
                <td>BMI</td>
                <td>{df_wide_full['BodyMassIndex'].iloc[-1]:.1f}</td>
                <td>{"✅ Normal (18.5-24.9)" if 18.5 <= df_wide_full['BodyMassIndex'].iloc[-1] <= 24.9 else "Note: BMI doesn't account for muscle"}</td>
            </tr>
            <tr>
                <td>Phase Angle (Avg)</td>
                <td>{df_wide_full['PhaseAngle_Avg'].iloc[-1]:.1f}°</td>
                <td>{"✅ Excellent (>5.0°)" if df_wide_full['PhaseAngle_Avg'].iloc[-1] > 5.0 else "Good"}</td>
            </tr>
        </table>

        <div class="footer">
            <p>This report was generated by Body Recomposition Tracker - Complete Version</p>
            <p>Data source: {current_filepath}</p>
            <p>To convert to PDF: Use your browser's Print function and select "Save as PDF"</p>
        </div>
    </body>
    </html>
    """

    # Return as downloadable HTML file
    return dict(
        content=html_content,
        filename=f"body_recomp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )


if __name__ == '__main__':
    app.run(debug=True, port=8052)
