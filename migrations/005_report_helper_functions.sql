-- Migration 005: Report Helper Functions
-- Creates RPC functions to support clinical report generation

-- Function to retrieve themed session chunks for analysis
CREATE OR REPLACE FUNCTION get_themed_chunks_for_analysis(
    p_patient_id TEXT,
    p_start_date TIMESTAMP WITH TIME ZONE,
    p_end_date TIMESTAMP WITH TIME ZONE
)
RETURNS TABLE (
    chunk_id TEXT,
    session_id TEXT,
    chunk_text TEXT,
    embedding VECTOR(1536),
    primary_emotion TEXT,
    polarity FLOAT,
    intensity FLOAT,
    psychological_theme TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
) 
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Validate that the user has access to this patient's data
    IF NOT EXISTS (
        SELECT 1 FROM validate_patient_access(auth.uid()::text, p_patient_id)
    ) THEN
        RAISE EXCEPTION 'Access denied to patient data';
    END IF;

    RETURN QUERY
    SELECT 
        sc.chunk_id,
        sc.session_id,
        sc.chunk_text,
        sc.embedding,
        sc.primary_emotion,
        sc.polarity,
        sc.intensity,
        sc.psychological_theme,
        sc.created_at,
        sc.metadata
    FROM session_chunks sc
    JOIN sessions s ON sc.session_id = s.session_id
    WHERE s.patient_id = p_patient_id
        AND sc.created_at >= p_start_date
        AND sc.created_at <= p_end_date
        AND sc.embedding IS NOT NULL
        AND sc.psychological_theme IS NOT NULL
    ORDER BY sc.created_at DESC;
END;
$$;

-- Function to count patient data points in a timeframe
CREATE OR REPLACE FUNCTION count_patient_data_points(
    p_patient_id TEXT,
    p_start_date TIMESTAMP WITH TIME ZONE,
    p_end_date TIMESTAMP WITH TIME ZONE
)
RETURNS TABLE (count BIGINT)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Validate that the user has access to this patient's data
    IF NOT EXISTS (
        SELECT 1 FROM validate_patient_access(auth.uid()::text, p_patient_id)
    ) THEN
        RAISE EXCEPTION 'Access denied to patient data';
    END IF;

    RETURN QUERY
    SELECT COUNT(*)::BIGINT
    FROM session_chunks sc
    JOIN sessions s ON sc.session_id = s.session_id
    WHERE s.patient_id = p_patient_id
        AND sc.created_at >= p_start_date
        AND sc.created_at <= p_end_date;
END;
$$;

-- Function to count patient sessions in a timeframe
CREATE OR REPLACE FUNCTION count_patient_sessions(
    p_patient_id TEXT,
    p_start_date TIMESTAMP WITH TIME ZONE,
    p_end_date TIMESTAMP WITH TIME ZONE
)
RETURNS TABLE (count BIGINT)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Validate that the user has access to this patient's data
    IF NOT EXISTS (
        SELECT 1 FROM validate_patient_access(auth.uid()::text, p_patient_id)
    ) THEN
        RAISE EXCEPTION 'Access denied to patient data';
    END IF;

    RETURN QUERY
    SELECT COUNT(DISTINCT s.session_id)::BIGINT
    FROM sessions s
    WHERE s.patient_id = p_patient_id
        AND s.created_at >= p_start_date
        AND s.created_at <= p_end_date;
END;
$$;

-- Function to get consolidated patient metrics for reporting
CREATE OR REPLACE FUNCTION get_consolidated_patient_metrics(
    p_patient_id TEXT,
    p_timeframe TEXT DEFAULT 'month'
)
RETURNS TABLE (
    total_sessions BIGINT,
    total_chunks BIGINT,
    avg_session_length FLOAT,
    avg_polarity FLOAT,
    avg_intensity FLOAT,
    most_common_emotion TEXT,
    most_common_theme TEXT,
    active_alerts_count BIGINT,
    first_session_date TIMESTAMP WITH TIME ZONE,
    last_session_date TIMESTAMP WITH TIME ZONE,
    engagement_score FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    start_date TIMESTAMP WITH TIME ZONE;
    end_date TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Validate that the user has access to this patient's data
    IF NOT EXISTS (
        SELECT 1 FROM validate_patient_access(auth.uid()::text, p_patient_id)
    ) THEN
        RAISE EXCEPTION 'Access denied to patient data';
    END IF;

    -- Calculate date range based on timeframe
    end_date := NOW();
    CASE p_timeframe
        WHEN 'week' THEN start_date := end_date - INTERVAL '1 week';
        WHEN 'month' THEN start_date := end_date - INTERVAL '1 month';
        WHEN 'quarter' THEN start_date := end_date - INTERVAL '3 months';
        WHEN 'year' THEN start_date := end_date - INTERVAL '1 year';
        ELSE start_date := end_date - INTERVAL '1 month';
    END CASE;

    RETURN QUERY
    WITH session_stats AS (
        SELECT 
            COUNT(DISTINCT s.session_id) as session_count,
            MIN(s.created_at) as first_session,
            MAX(s.created_at) as last_session,
            AVG(EXTRACT(EPOCH FROM (s.updated_at - s.created_at))/60) as avg_length_minutes
        FROM sessions s
        WHERE s.patient_id = p_patient_id
            AND s.created_at >= start_date
            AND s.created_at <= end_date
    ),
    chunk_stats AS (
        SELECT 
            COUNT(*) as chunk_count,
            AVG(sc.polarity) as avg_pol,
            AVG(sc.intensity) as avg_int,
            MODE() WITHIN GROUP (ORDER BY sc.primary_emotion) as common_emotion,
            MODE() WITHIN GROUP (ORDER BY sc.psychological_theme) as common_theme
        FROM session_chunks sc
        JOIN sessions s ON sc.session_id = s.session_id
        WHERE s.patient_id = p_patient_id
            AND sc.created_at >= start_date
            AND sc.created_at <= end_date
    ),
    alert_stats AS (
        SELECT COUNT(*) as alert_count
        FROM clinical_alerts ca
        WHERE ca.patient_id = p_patient_id
            AND ca.is_active = true
            AND ca.created_at >= start_date
    )
    SELECT 
        ss.session_count,
        cs.chunk_count,
        ss.avg_length_minutes,
        cs.avg_pol,
        cs.avg_int,
        cs.common_emotion,
        cs.common_theme,
        als.alert_count,
        ss.first_session,
        ss.last_session,
        -- Engagement score based on session frequency and length
        CASE 
            WHEN ss.session_count = 0 THEN 0.0
            ELSE LEAST(1.0, (ss.session_count::FLOAT / 30.0) * (ss.avg_length_minutes / 15.0))
        END as engagement
    FROM session_stats ss, chunk_stats cs, alert_stats als;
END;
$$;

-- Grant execute permissions to appropriate roles
GRANT EXECUTE ON FUNCTION get_themed_chunks_for_analysis(TEXT, TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE) TO psychologist, admin;
GRANT EXECUTE ON FUNCTION count_patient_data_points(TEXT, TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE) TO psychologist, admin;
GRANT EXECUTE ON FUNCTION count_patient_sessions(TEXT, TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE) TO psychologist, admin;
GRANT EXECUTE ON FUNCTION get_consolidated_patient_metrics(TEXT, TEXT) TO psychologist, admin;

-- Create indexes to optimize report queries
CREATE INDEX IF NOT EXISTS idx_session_chunks_patient_theme_date 
ON session_chunks USING btree (
    (SELECT patient_id FROM sessions WHERE sessions.session_id = session_chunks.session_id),
    psychological_theme,
    created_at
);

CREATE INDEX IF NOT EXISTS idx_session_chunks_embedding_theme 
ON session_chunks USING ivfflat (embedding vector_cosine_ops) 
WHERE embedding IS NOT NULL AND psychological_theme IS NOT NULL;

-- Add comment for migration tracking
COMMENT ON FUNCTION get_themed_chunks_for_analysis IS 'Migration 005: Retrieves session chunks with theme metadata for clinical report analysis';
COMMENT ON FUNCTION count_patient_data_points IS 'Migration 005: Counts patient data points in specified timeframe for reporting';
COMMENT ON FUNCTION count_patient_sessions IS 'Migration 005: Counts patient sessions in specified timeframe for reporting';
COMMENT ON FUNCTION get_consolidated_patient_metrics IS 'Migration 005: Provides consolidated patient metrics for clinical reports';
