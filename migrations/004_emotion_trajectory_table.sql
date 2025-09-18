-- Migration 004: Emotion Trajectory Tracking Table
-- Creates dedicated table for emotional trajectory data points and clinical alerts
-- Supports real-time emotional monitoring and clinical safety features

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create emotion_trajectory table for storing emotional data points
CREATE TABLE IF NOT EXISTS emotion_trajectory (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES session_chunks(id) ON DELETE SET NULL,
    
    -- Temporal data
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Emotional metadata
    emotion VARCHAR(20) NOT NULL CHECK (emotion IN ('happy', 'sad', 'angry', 'fear', 'neutral', 'anxiety', 'depression')),
    intensity DECIMAL(3,2) NOT NULL CHECK (intensity >= 0.0 AND intensity <= 1.0),
    polarity DECIMAL(3,2) NOT NULL CHECK (polarity >= -1.0 AND polarity <= 1.0),
    
    -- Contextual data
    theme TEXT,
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create clinical_alerts table for automated clinical safety alerts
CREATE TABLE IF NOT EXISTS clinical_alerts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    -- Alert metadata
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    message TEXT NOT NULL,
    
    -- Alert data
    threshold_exceeded DECIMAL(5,3),
    metadata JSONB DEFAULT '{}',
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved')),
    acknowledged_by UUID REFERENCES patients(id) ON DELETE SET NULL,
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_emotion_trajectory_patient_timestamp ON emotion_trajectory(patient_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_emotion_trajectory_session ON emotion_trajectory(session_id);
CREATE INDEX IF NOT EXISTS idx_emotion_trajectory_emotion ON emotion_trajectory(emotion);
CREATE INDEX IF NOT EXISTS idx_emotion_trajectory_polarity ON emotion_trajectory(polarity);
CREATE INDEX IF NOT EXISTS idx_emotion_trajectory_timestamp ON emotion_trajectory(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_clinical_alerts_patient ON clinical_alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_clinical_alerts_severity ON clinical_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_clinical_alerts_status ON clinical_alerts(status);
CREATE INDEX IF NOT EXISTS idx_clinical_alerts_created ON clinical_alerts(created_at DESC);

-- Enable Row Level Security
ALTER TABLE emotion_trajectory ENABLE ROW LEVEL SECURITY;
ALTER TABLE clinical_alerts ENABLE ROW LEVEL SECURITY;

-- RLS Policies for emotion_trajectory table

-- Patients can only view their own emotional trajectory data
CREATE POLICY "emotion_trajectory_patient_select" ON emotion_trajectory
    FOR SELECT
    USING (
        auth.uid() = patient_id AND 
        current_setting('app.user_role', true) = 'patient'
    );

-- Psychologists can view trajectory data for assigned patients only
CREATE POLICY "emotion_trajectory_psychologist_select" ON emotion_trajectory
    FOR SELECT
    USING (
        current_setting('app.user_role', true) = 'psychologist' AND
        patient_id IN (
            SELECT patient_id FROM assignments 
            WHERE psychologist_id = auth.uid() AND status = 'active'
        )
    );

-- Admins have full access to all trajectory data
CREATE POLICY "emotion_trajectory_admin_all" ON emotion_trajectory
    FOR ALL
    USING (current_setting('app.user_role', true) = 'admin');

-- System can insert trajectory data for any patient (for automated recording)
CREATE POLICY "emotion_trajectory_system_insert" ON emotion_trajectory
    FOR INSERT
    WITH CHECK (
        current_setting('app.user_role', true) IN ('system', 'psychologist', 'admin')
    );

-- RLS Policies for clinical_alerts table

-- Patients can view their own alerts
CREATE POLICY "clinical_alerts_patient_select" ON clinical_alerts
    FOR SELECT
    USING (
        auth.uid() = patient_id AND 
        current_setting('app.user_role', true) = 'patient'
    );

-- Patients can acknowledge their own alerts
CREATE POLICY "clinical_alerts_patient_update" ON clinical_alerts
    FOR UPDATE
    USING (
        auth.uid() = patient_id AND 
        current_setting('app.user_role', true) = 'patient'
    )
    WITH CHECK (
        auth.uid() = patient_id AND 
        current_setting('app.user_role', true) = 'patient'
    );

-- Psychologists can view and manage alerts for assigned patients
CREATE POLICY "clinical_alerts_psychologist_select" ON clinical_alerts
    FOR SELECT
    USING (
        current_setting('app.user_role', true) = 'psychologist' AND
        patient_id IN (
            SELECT patient_id FROM assignments 
            WHERE psychologist_id = auth.uid() AND status = 'active'
        )
    );

CREATE POLICY "clinical_alerts_psychologist_update" ON clinical_alerts
    FOR UPDATE
    USING (
        current_setting('app.user_role', true) = 'psychologist' AND
        patient_id IN (
            SELECT patient_id FROM assignments 
            WHERE psychologist_id = auth.uid() AND status = 'active'
        )
    )
    WITH CHECK (
        current_setting('app.user_role', true) = 'psychologist' AND
        patient_id IN (
            SELECT patient_id FROM assignments 
            WHERE psychologist_id = auth.uid() AND status = 'active'
        )
    );

-- Admins have full access to all alerts
CREATE POLICY "clinical_alerts_admin_all" ON clinical_alerts
    FOR ALL
    USING (current_setting('app.user_role', true) = 'admin');

-- System can create alerts for any patient
CREATE POLICY "clinical_alerts_system_insert" ON clinical_alerts
    FOR INSERT
    WITH CHECK (
        current_setting('app.user_role', true) IN ('system', 'psychologist', 'admin')
    );

-- Create RPC function to store emotion trajectory points
CREATE OR REPLACE FUNCTION store_emotion_trajectory_point(
    p_patient_id UUID,
    p_timestamp TIMESTAMPTZ,
    p_emotion VARCHAR(20),
    p_intensity DECIMAL(3,2),
    p_polarity DECIMAL(3,2),
    p_session_id UUID,
    p_chunk_id UUID DEFAULT NULL,
    p_theme TEXT DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    trajectory_id UUID;
BEGIN
    -- Validate emotion type
    IF p_emotion NOT IN ('happy', 'sad', 'angry', 'fear', 'neutral', 'anxiety', 'depression') THEN
        RAISE EXCEPTION 'Invalid emotion type: %', p_emotion;
    END IF;
    
    -- Validate ranges
    IF p_intensity < 0.0 OR p_intensity > 1.0 THEN
        RAISE EXCEPTION 'Intensity must be between 0.0 and 1.0, got: %', p_intensity;
    END IF;
    
    IF p_polarity < -1.0 OR p_polarity > 1.0 THEN
        RAISE EXCEPTION 'Polarity must be between -1.0 and 1.0, got: %', p_polarity;
    END IF;
    
    -- Insert trajectory point
    INSERT INTO emotion_trajectory (
        patient_id, timestamp, emotion, intensity, polarity, 
        session_id, chunk_id, theme
    ) VALUES (
        p_patient_id, p_timestamp, p_emotion, p_intensity, p_polarity,
        p_session_id, p_chunk_id, p_theme
    ) RETURNING id INTO trajectory_id;
    
    RETURN trajectory_id;
END;
$$;

-- Create RPC function to get emotion trajectory data
CREATE OR REPLACE FUNCTION get_emotion_trajectory_data(
    p_patient_id UUID,
    p_start_date TIMESTAMPTZ,
    p_end_date TIMESTAMPTZ
) RETURNS TABLE (
    id UUID,
    timestamp TIMESTAMPTZ,
    emotion VARCHAR(20),
    intensity DECIMAL(3,2),
    polarity DECIMAL(3,2),
    session_id UUID,
    chunk_id UUID,
    theme TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        et.id,
        et.timestamp,
        et.emotion,
        et.intensity,
        et.polarity,
        et.session_id,
        et.chunk_id,
        et.theme
    FROM emotion_trajectory et
    WHERE et.patient_id = p_patient_id
      AND et.timestamp >= p_start_date
      AND et.timestamp <= p_end_date
    ORDER BY et.timestamp ASC;
END;
$$;

-- Create RPC function to log clinical alerts
CREATE OR REPLACE FUNCTION log_clinical_alert(
    p_patient_id UUID,
    p_alert_type VARCHAR(50),
    p_severity VARCHAR(20),
    p_message TEXT,
    p_threshold_exceeded DECIMAL(5,3) DEFAULT NULL,
    p_data_points INTEGER DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    alert_id UUID;
    alert_metadata JSONB;
BEGIN
    -- Validate severity
    IF p_severity NOT IN ('low', 'medium', 'high', 'critical') THEN
        RAISE EXCEPTION 'Invalid severity level: %', p_severity;
    END IF;
    
    -- Build metadata
    alert_metadata := jsonb_build_object(
        'threshold_exceeded', p_threshold_exceeded,
        'data_points_analyzed', p_data_points,
        'detection_timestamp', NOW()
    );
    
    -- Insert alert
    INSERT INTO clinical_alerts (
        patient_id, alert_type, severity, message, 
        threshold_exceeded, metadata
    ) VALUES (
        p_patient_id, p_alert_type, p_severity, p_message,
        p_threshold_exceeded, alert_metadata
    ) RETURNING id INTO alert_id;
    
    RETURN alert_id;
END;
$$;

-- Create RPC function to create high-severity clinical alerts
CREATE OR REPLACE FUNCTION create_clinical_alert(
    p_patient_id UUID,
    p_alert_type VARCHAR(50),
    p_severity VARCHAR(20),
    p_message TEXT,
    p_metadata JSONB DEFAULT '{}'
) RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    alert_id UUID;
BEGIN
    -- Validate severity
    IF p_severity NOT IN ('low', 'medium', 'high', 'critical') THEN
        RAISE EXCEPTION 'Invalid severity level: %', p_severity;
    END IF;
    
    -- Insert alert
    INSERT INTO clinical_alerts (
        patient_id, alert_type, severity, message, metadata
    ) VALUES (
        p_patient_id, p_alert_type, p_severity, p_message, p_metadata
    ) RETURNING id INTO alert_id;
    
    RETURN alert_id;
END;
$$;

-- Create RPC function to get active clinical alerts
CREATE OR REPLACE FUNCTION get_active_clinical_alerts(
    p_patient_id UUID DEFAULT NULL,
    p_severity VARCHAR(20) DEFAULT NULL
) RETURNS TABLE (
    id UUID,
    patient_id UUID,
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    message TEXT,
    threshold_exceeded DECIMAL(5,3),
    metadata JSONB,
    status VARCHAR(20),
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ca.id,
        ca.patient_id,
        ca.alert_type,
        ca.severity,
        ca.message,
        ca.threshold_exceeded,
        ca.metadata,
        ca.status,
        ca.created_at
    FROM clinical_alerts ca
    WHERE ca.status = 'active'
      AND (p_patient_id IS NULL OR ca.patient_id = p_patient_id)
      AND (p_severity IS NULL OR ca.severity = p_severity)
    ORDER BY ca.created_at DESC;
END;
$$;

-- Create RPC function to acknowledge clinical alerts
CREATE OR REPLACE FUNCTION acknowledge_clinical_alert(
    p_alert_id UUID,
    p_acknowledged_by UUID
) RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    UPDATE clinical_alerts
    SET 
        status = 'acknowledged',
        acknowledged_by = p_acknowledged_by,
        acknowledged_at = NOW(),
        updated_at = NOW()
    WHERE id = p_alert_id
      AND status = 'active';
    
    RETURN FOUND;
END;
$$;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_emotion_trajectory_updated_at
    BEFORE UPDATE ON emotion_trajectory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clinical_alerts_updated_at
    BEFORE UPDATE ON clinical_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant execute permissions on RPC functions
GRANT EXECUTE ON FUNCTION store_emotion_trajectory_point TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION get_emotion_trajectory_data TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION log_clinical_alert TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION create_clinical_alert TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION get_active_clinical_alerts TO patient, psychologist, admin;
GRANT EXECUTE ON FUNCTION acknowledge_clinical_alert TO patient, psychologist, admin;

-- Grant table permissions
GRANT SELECT ON emotion_trajectory TO patient, psychologist, admin;
GRANT INSERT ON emotion_trajectory TO psychologist, admin;
GRANT UPDATE ON emotion_trajectory TO admin;
GRANT DELETE ON emotion_trajectory TO admin;

GRANT SELECT ON clinical_alerts TO patient, psychologist, admin;
GRANT INSERT ON clinical_alerts TO psychologist, admin;
GRANT UPDATE ON clinical_alerts TO patient, psychologist, admin;
GRANT DELETE ON clinical_alerts TO admin;

-- Add helpful comments
COMMENT ON TABLE emotion_trajectory IS 'Stores emotional trajectory data points for therapeutic monitoring';
COMMENT ON TABLE clinical_alerts IS 'Stores automated clinical safety alerts based on emotional patterns';

COMMENT ON COLUMN emotion_trajectory.emotion IS 'Primary emotion type (happy, sad, angry, fear, neutral, anxiety, depression)';
COMMENT ON COLUMN emotion_trajectory.intensity IS 'Emotion intensity from 0.0 (weak) to 1.0 (strong)';
COMMENT ON COLUMN emotion_trajectory.polarity IS 'Emotion polarity from -1.0 (negative) to 1.0 (positive)';

COMMENT ON COLUMN clinical_alerts.severity IS 'Alert severity level (low, medium, high, critical)';
COMMENT ON COLUMN clinical_alerts.threshold_exceeded IS 'Numerical threshold that triggered the alert';
COMMENT ON COLUMN clinical_alerts.status IS 'Alert status (active, acknowledged, resolved)';
