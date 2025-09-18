-- Migration: Create layered memory architecture tables
-- Creates short_term_memory, medium_term_memory, and long_term_memory tables
-- with proper RLS policies for therapeutic data security

-- Create short_term_memory table
CREATE TABLE IF NOT EXISTS short_term_memory (
    id SERIAL PRIMARY KEY,
    patient_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384), -- Default embedding dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days') -- Auto-expire after 7 days
);

-- Create medium_term_memory table
CREATE TABLE IF NOT EXISTS medium_term_memory (
    id SERIAL PRIMARY KEY,
    patient_id TEXT NOT NULL,
    session_id TEXT, -- Can be NULL for consolidated memories
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    week_start DATE NOT NULL, -- Week this memory represents
    consolidation_source TEXT DEFAULT 'weekly_clustering', -- How this memory was created
    source_memory_count INTEGER DEFAULT 1 -- Number of source memories consolidated
);

-- Create long_term_memory table
CREATE TABLE IF NOT EXISTS long_term_memory (
    id SERIAL PRIMARY KEY,
    patient_id TEXT NOT NULL,
    session_id TEXT, -- Can be NULL for theme-based memories
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    theme_category TEXT NOT NULL, -- Primary psychological theme
    recurrence_count INTEGER DEFAULT 1, -- How many times this theme appeared
    first_occurrence DATE, -- When this theme first appeared
    last_occurrence DATE, -- Most recent occurrence
    consolidation_source TEXT DEFAULT 'theme_clustering' -- How this memory was created
);

-- Create indexes for performance
-- Short-term memory indexes
CREATE INDEX IF NOT EXISTS idx_short_term_memory_patient_id ON short_term_memory(patient_id);
CREATE INDEX IF NOT EXISTS idx_short_term_memory_session_id ON short_term_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_short_term_memory_created_at ON short_term_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_short_term_memory_expires_at ON short_term_memory(expires_at);
CREATE INDEX IF NOT EXISTS idx_short_term_memory_embedding ON short_term_memory 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_short_term_memory_metadata ON short_term_memory USING GIN (metadata);

-- Medium-term memory indexes
CREATE INDEX IF NOT EXISTS idx_medium_term_memory_patient_id ON medium_term_memory(patient_id);
CREATE INDEX IF NOT EXISTS idx_medium_term_memory_week_start ON medium_term_memory(week_start);
CREATE INDEX IF NOT EXISTS idx_medium_term_memory_created_at ON medium_term_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_medium_term_memory_embedding ON medium_term_memory 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_medium_term_memory_metadata ON medium_term_memory USING GIN (metadata);

-- Long-term memory indexes
CREATE INDEX IF NOT EXISTS idx_long_term_memory_patient_id ON long_term_memory(patient_id);
CREATE INDEX IF NOT EXISTS idx_long_term_memory_theme_category ON long_term_memory(theme_category);
CREATE INDEX IF NOT EXISTS idx_long_term_memory_first_occurrence ON long_term_memory(first_occurrence);
CREATE INDEX IF NOT EXISTS idx_long_term_memory_last_occurrence ON long_term_memory(last_occurrence);
CREATE INDEX IF NOT EXISTS idx_long_term_memory_embedding ON long_term_memory 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_long_term_memory_metadata ON long_term_memory USING GIN (metadata);

-- Create RLS policies
-- Enable RLS on all memory tables
ALTER TABLE short_term_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE medium_term_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE long_term_memory ENABLE ROW LEVEL SECURITY;

-- Short-term memory RLS policies
CREATE POLICY "Patients can access only their short-term memories" ON short_term_memory
    FOR ALL USING (patient_id = current_setting('app.current_patient_id', true));

CREATE POLICY "Psychologists can access their patients' short-term memories" ON short_term_memory
    FOR ALL USING (
        patient_id IN (
            SELECT patient_id FROM patient_psychologist_assignments 
            WHERE psychologist_id = current_setting('app.current_psychologist_id', true)
        )
    );

CREATE POLICY "Admins can access all short-term memories" ON short_term_memory
    FOR ALL USING (current_setting('app.current_user_role', true) = 'admin');

-- Medium-term memory RLS policies
CREATE POLICY "Patients can access only their medium-term memories" ON medium_term_memory
    FOR ALL USING (patient_id = current_setting('app.current_patient_id', true));

CREATE POLICY "Psychologists can access their patients' medium-term memories" ON medium_term_memory
    FOR ALL USING (
        patient_id IN (
            SELECT patient_id FROM patient_psychologist_assignments 
            WHERE psychologist_id = current_setting('app.current_psychologist_id', true)
        )
    );

CREATE POLICY "Admins can access all medium-term memories" ON medium_term_memory
    FOR ALL USING (current_setting('app.current_user_role', true) = 'admin');

-- Long-term memory RLS policies
CREATE POLICY "Patients can access only their long-term memories" ON long_term_memory
    FOR ALL USING (patient_id = current_setting('app.current_patient_id', true));

CREATE POLICY "Psychologists can access their patients' long-term memories" ON long_term_memory
    FOR ALL USING (
        patient_id IN (
            SELECT patient_id FROM patient_psychologist_assignments 
            WHERE psychologist_id = current_setting('app.current_psychologist_id', true)
        )
    );

CREATE POLICY "Admins can access all long-term memories" ON long_term_memory
    FOR ALL USING (current_setting('app.current_user_role', true) = 'admin');

-- Create functions for memory operations
-- Function to store short-term memory
CREATE OR REPLACE FUNCTION store_short_term_memory(
    p_schema_name TEXT,
    p_patient_id TEXT,
    p_session_id TEXT,
    p_content TEXT,
    p_embedding vector(384),
    p_metadata JSONB DEFAULT '{}'
) RETURNS INTEGER AS $$
DECLARE
    memory_id INTEGER;
    table_name TEXT;
BEGIN
    table_name := p_schema_name || '.short_term_memory';
    
    EXECUTE format('
        INSERT INTO %I (patient_id, session_id, content, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
    ', table_name)
    USING p_patient_id, p_session_id, p_content, p_embedding, p_metadata
    INTO memory_id;
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to store medium-term memory
CREATE OR REPLACE FUNCTION store_medium_term_memory(
    p_schema_name TEXT,
    p_patient_id TEXT,
    p_session_id TEXT,
    p_content TEXT,
    p_embedding vector(384),
    p_metadata JSONB DEFAULT '{}',
    p_week_start DATE DEFAULT NULL,
    p_consolidation_source TEXT DEFAULT 'weekly_clustering',
    p_source_memory_count INTEGER DEFAULT 1
) RETURNS INTEGER AS $$
DECLARE
    memory_id INTEGER;
    table_name TEXT;
    week_start_date DATE;
BEGIN
    table_name := p_schema_name || '.medium_term_memory';
    
    -- Calculate week start if not provided
    week_start_date := COALESCE(p_week_start, date_trunc('week', CURRENT_DATE)::DATE);
    
    EXECUTE format('
        INSERT INTO %I (patient_id, session_id, content, embedding, metadata, week_start, consolidation_source, source_memory_count)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
    ', table_name)
    USING p_patient_id, p_session_id, p_content, p_embedding, p_metadata, week_start_date, p_consolidation_source, p_source_memory_count
    INTO memory_id;
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to store long-term memory
CREATE OR REPLACE FUNCTION store_long_term_memory(
    p_schema_name TEXT,
    p_patient_id TEXT,
    p_session_id TEXT,
    p_content TEXT,
    p_embedding vector(384),
    p_metadata JSONB DEFAULT '{}',
    p_theme_category TEXT DEFAULT 'general',
    p_recurrence_count INTEGER DEFAULT 1,
    p_first_occurrence DATE DEFAULT NULL,
    p_last_occurrence DATE DEFAULT NULL,
    p_consolidation_source TEXT DEFAULT 'theme_clustering'
) RETURNS INTEGER AS $$
DECLARE
    memory_id INTEGER;
    table_name TEXT;
    first_occ DATE;
    last_occ DATE;
BEGIN
    table_name := p_schema_name || '.long_term_memory';
    
    -- Set default dates if not provided
    first_occ := COALESCE(p_first_occurrence, CURRENT_DATE);
    last_occ := COALESCE(p_last_occurrence, CURRENT_DATE);
    
    EXECUTE format('
        INSERT INTO %I (patient_id, session_id, content, embedding, metadata, theme_category, recurrence_count, first_occurrence, last_occurrence, consolidation_source)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id
    ', table_name)
    USING p_patient_id, p_session_id, p_content, p_embedding, p_metadata, p_theme_category, p_recurrence_count, first_occ, last_occ, p_consolidation_source
    INTO memory_id;
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to retrieve memories by similarity
CREATE OR REPLACE FUNCTION retrieve_memories_by_similarity(
    p_schema_name TEXT,
    p_memory_table TEXT, -- 'short_term_memory', 'medium_term_memory', or 'long_term_memory'
    p_query_embedding vector(384),
    p_patient_id TEXT,
    p_session_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 5,
    p_min_similarity FLOAT DEFAULT 0.1,
    p_metadata_filters JSONB DEFAULT NULL
) RETURNS TABLE (
    id INTEGER,
    patient_id TEXT,
    session_id TEXT,
    content TEXT,
    embedding vector(384),
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    table_name TEXT;
    query_text TEXT;
    where_conditions TEXT[];
    param_count INTEGER := 3; -- Starting parameter count
BEGIN
    table_name := p_schema_name || '.' || p_memory_table;
    
    -- Build base query
    query_text := format('
        SELECT 
            m.id,
            m.patient_id,
            m.session_id,
            m.content,
            m.embedding,
            m.metadata,
            1 - (m.embedding <=> $1) as similarity,
            m.created_at
        FROM %I m
        WHERE m.patient_id = $2
        AND 1 - (m.embedding <=> $1) >= $3
    ', table_name);
    
    -- Add session filter if provided
    IF p_session_id IS NOT NULL THEN
        param_count := param_count + 1;
        query_text := query_text || format(' AND m.session_id = $%s', param_count);
    END IF;
    
    -- Add metadata filters if provided
    IF p_metadata_filters IS NOT NULL THEN
        param_count := param_count + 1;
        query_text := query_text || format(' AND m.metadata @> $%s', param_count);
    END IF;
    
    -- Add ordering and limit
    query_text := query_text || format(' ORDER BY m.embedding <=> $1 LIMIT $%s', param_count + 1);
    
    -- Execute query based on parameters
    IF p_session_id IS NOT NULL AND p_metadata_filters IS NOT NULL THEN
        RETURN QUERY EXECUTE query_text USING p_query_embedding, p_patient_id, p_min_similarity, p_session_id, p_metadata_filters, p_limit;
    ELSIF p_session_id IS NOT NULL THEN
        RETURN QUERY EXECUTE query_text USING p_query_embedding, p_patient_id, p_min_similarity, p_session_id, p_limit;
    ELSIF p_metadata_filters IS NOT NULL THEN
        RETURN QUERY EXECUTE query_text USING p_query_embedding, p_patient_id, p_min_similarity, p_metadata_filters, p_limit;
    ELSE
        RETURN QUERY EXECUTE query_text USING p_query_embedding, p_patient_id, p_min_similarity, p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get memory statistics
CREATE OR REPLACE FUNCTION get_memory_layer_stats(
    p_schema_name TEXT,
    p_memory_table TEXT,
    p_patient_id TEXT
) RETURNS TABLE (
    total_memories INTEGER,
    oldest_memory TIMESTAMP WITH TIME ZONE,
    newest_memory TIMESTAMP WITH TIME ZONE,
    avg_content_length FLOAT,
    unique_sessions INTEGER,
    metadata_summary JSONB
) AS $$
DECLARE
    table_name TEXT;
BEGIN
    table_name := p_schema_name || '.' || p_memory_table;
    
    RETURN QUERY EXECUTE format('
        SELECT 
            COUNT(*)::INTEGER as total_memories,
            MIN(created_at) as oldest_memory,
            MAX(created_at) as newest_memory,
            AVG(LENGTH(content))::FLOAT as avg_content_length,
            COUNT(DISTINCT session_id)::INTEGER as unique_sessions,
            jsonb_object_agg(
                COALESCE(metadata->>''primary_emotion'', ''unknown''),
                emotion_count
            ) as metadata_summary
        FROM (
            SELECT 
                created_at,
                content,
                session_id,
                metadata,
                COUNT(*) as emotion_count
            FROM %I
            WHERE patient_id = $1
            GROUP BY created_at, content, session_id, metadata->>''primary_emotion'', metadata
        ) grouped
    ', table_name)
    USING p_patient_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to clean expired short-term memories
CREATE OR REPLACE FUNCTION cleanup_expired_short_term_memories(
    p_schema_name TEXT
) RETURNS INTEGER AS $$
DECLARE
    table_name TEXT;
    deleted_count INTEGER;
BEGIN
    table_name := p_schema_name || '.short_term_memory';
    
    EXECUTE format('
        DELETE FROM %I 
        WHERE expires_at < NOW()
    ', table_name);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_memory_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all memory tables
CREATE TRIGGER trigger_update_short_term_memory_updated_at
    BEFORE UPDATE ON short_term_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_updated_at();

CREATE TRIGGER trigger_update_medium_term_memory_updated_at
    BEFORE UPDATE ON medium_term_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_updated_at();

CREATE TRIGGER trigger_update_long_term_memory_updated_at
    BEFORE UPDATE ON long_term_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_updated_at();

-- Create schema-specific table creation functions
CREATE OR REPLACE FUNCTION create_memory_tables_in_schema(schema_name TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    table_names TEXT[] := ARRAY['short_term_memory', 'medium_term_memory', 'long_term_memory'];
    table_name TEXT;
    full_table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY table_names
    LOOP
        full_table_name := schema_name || '.' || table_name;
        
        -- Create table based on type
        IF table_name = 'short_term_memory' THEN
            EXECUTE format('
                CREATE TABLE IF NOT EXISTS %I (
                    id SERIAL PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB DEFAULT ''{}''::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL ''7 days'')
                )', full_table_name);
                
        ELSIF table_name = 'medium_term_memory' THEN
            EXECUTE format('
                CREATE TABLE IF NOT EXISTS %I (
                    id SERIAL PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB DEFAULT ''{}''::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    week_start DATE NOT NULL,
                    consolidation_source TEXT DEFAULT ''weekly_clustering'',
                    source_memory_count INTEGER DEFAULT 1
                )', full_table_name);
                
        ELSIF table_name = 'long_term_memory' THEN
            EXECUTE format('
                CREATE TABLE IF NOT EXISTS %I (
                    id SERIAL PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB DEFAULT ''{}''::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    theme_category TEXT NOT NULL,
                    recurrence_count INTEGER DEFAULT 1,
                    first_occurrence DATE,
                    last_occurrence DATE,
                    consolidation_source TEXT DEFAULT ''theme_clustering''
                )', full_table_name);
        END IF;
        
        -- Create indexes
        EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I(patient_id)', 
                       'idx_' || schema_name || '_' || table_name || '_patient_id', full_table_name);
        EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I(created_at)', 
                       'idx_' || schema_name || '_' || table_name || '_created_at', full_table_name);
        EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)', 
                       'idx_' || schema_name || '_' || table_name || '_embedding', full_table_name);
        EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING GIN (metadata)', 
                       'idx_' || schema_name || '_' || table_name || '_metadata', full_table_name);
        
        -- Create trigger
        EXECUTE format('
            CREATE TRIGGER %I
                BEFORE UPDATE ON %I
                FOR EACH ROW
                EXECUTE FUNCTION update_memory_updated_at()',
            'trigger_update_' || schema_name || '_' || table_name || '_updated_at', full_table_name);
    END LOOP;
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Error creating memory tables in schema %: %', schema_name, SQLERRM;
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add comments for documentation
COMMENT ON TABLE short_term_memory IS 'Stores detailed chunks from the last active session only';
COMMENT ON TABLE medium_term_memory IS 'Stores weekly summaries from clustered embeddings of session chunks';
COMMENT ON TABLE long_term_memory IS 'Stores consolidated themes across multiple weeks with recurrent psychological themes';

COMMENT ON COLUMN short_term_memory.expires_at IS 'Automatic expiration timestamp for session data cleanup';
COMMENT ON COLUMN medium_term_memory.week_start IS 'Start date of the week this memory represents';
COMMENT ON COLUMN medium_term_memory.consolidation_source IS 'Method used to create this consolidated memory';
COMMENT ON COLUMN long_term_memory.theme_category IS 'Primary psychological theme category';
COMMENT ON COLUMN long_term_memory.recurrence_count IS 'Number of times this theme has appeared';
COMMENT ON COLUMN long_term_memory.first_occurrence IS 'Date when this theme first appeared';
COMMENT ON COLUMN long_term_memory.last_occurrence IS 'Most recent occurrence of this theme';
