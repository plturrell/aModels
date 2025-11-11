/**
 * Phase 2.1: Graph Filters Component
 * 
 * Advanced filtering for graph visualization including time-based filtering
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Button,
  Stack,
  Checkbox,
  FormControlLabel,
  DatePicker,
} from '@mui/material';
import ClearIcon from '@mui/icons-material/Clear';
import FilterListIcon from '@mui/icons-material/FilterList';

export interface GraphFilterState {
  nodeTypes: string[];
  edgeTypes: string[];
  dateFrom?: Date;
  dateTo?: Date;
  propertyFilters: Record<string, string>;
  minConnections?: number;
  maxConnections?: number;
}

export interface GraphFiltersProps {
  availableNodeTypes: string[];
  availableEdgeTypes: string[];
  onFilterChange: (filters: GraphFilterState) => void;
  initialFilters?: Partial<GraphFilterState>;
}

export function GraphFilters({
  availableNodeTypes,
  availableEdgeTypes,
  onFilterChange,
  initialFilters = {},
}: GraphFiltersProps) {
  const [nodeTypes, setNodeTypes] = useState<string[]>(initialFilters.nodeTypes || []);
  const [edgeTypes, setEdgeTypes] = useState<string[]>(initialFilters.edgeTypes || []);
  const [dateFrom, setDateFrom] = useState<Date | null>(initialFilters.dateFrom || null);
  const [dateTo, setDateTo] = useState<Date | null>(initialFilters.dateTo || null);
  const [minConnections, setMinConnections] = useState<number | ''>(initialFilters.minConnections || '');
  const [maxConnections, setMaxConnections] = useState<number | ''>(initialFilters.maxConnections || '');
  const [propertyKey, setPropertyKey] = useState('');
  const [propertyValue, setPropertyValue] = useState('');
  const [propertyFilters, setPropertyFilters] = useState<Record<string, string>>(
    initialFilters.propertyFilters || {}
  );

  const handleNodeTypeToggle = useCallback((type: string) => {
    setNodeTypes(prev => {
      const newTypes = prev.includes(type)
        ? prev.filter(t => t !== type)
        : [...prev, type];
      onFilterChange({
        nodeTypes: newTypes,
        edgeTypes,
        dateFrom: dateFrom || undefined,
        dateTo: dateTo || undefined,
        propertyFilters,
        minConnections: minConnections === '' ? undefined : minConnections,
        maxConnections: maxConnections === '' ? undefined : maxConnections,
      });
      return newTypes;
    });
  }, [edgeTypes, dateFrom, dateTo, propertyFilters, minConnections, maxConnections, onFilterChange]);

  const handleEdgeTypeToggle = useCallback((type: string) => {
    setEdgeTypes(prev => {
      const newTypes = prev.includes(type)
        ? prev.filter(t => t !== type)
        : [...prev, type];
      onFilterChange({
        nodeTypes,
        edgeTypes: newTypes,
        dateFrom: dateFrom || undefined,
        dateTo: dateTo || undefined,
        propertyFilters,
        minConnections: minConnections === '' ? undefined : minConnections,
        maxConnections: maxConnections === '' ? undefined : maxConnections,
      });
      return newTypes;
    });
  }, [nodeTypes, dateFrom, dateTo, propertyFilters, minConnections, maxConnections, onFilterChange]);

  const handleDateChange = useCallback((from: Date | null, to: Date | null) => {
    setDateFrom(from);
    setDateTo(to);
    onFilterChange({
      nodeTypes,
      edgeTypes,
      dateFrom: from || undefined,
      dateTo: to || undefined,
      propertyFilters,
      minConnections: minConnections === '' ? undefined : minConnections,
      maxConnections: maxConnections === '' ? undefined : maxConnections,
    });
  }, [nodeTypes, edgeTypes, propertyFilters, minConnections, maxConnections, onFilterChange]);

  const handleAddPropertyFilter = useCallback(() => {
    if (propertyKey && propertyValue) {
      const newFilters = { ...propertyFilters, [propertyKey]: propertyValue };
      setPropertyFilters(newFilters);
      setPropertyKey('');
      setPropertyValue('');
      onFilterChange({
        nodeTypes,
        edgeTypes,
        dateFrom: dateFrom || undefined,
        dateTo: dateTo || undefined,
        propertyFilters: newFilters,
        minConnections: minConnections === '' ? undefined : minConnections,
        maxConnections: maxConnections === '' ? undefined : maxConnections,
      });
    }
  }, [propertyKey, propertyValue, propertyFilters, nodeTypes, edgeTypes, dateFrom, dateTo, minConnections, maxConnections, onFilterChange]);

  const handleRemovePropertyFilter = useCallback((key: string) => {
    const newFilters = { ...propertyFilters };
    delete newFilters[key];
    setPropertyFilters(newFilters);
    onFilterChange({
      nodeTypes,
      edgeTypes,
      dateFrom: dateFrom || undefined,
      dateTo: dateTo || undefined,
      propertyFilters: newFilters,
      minConnections: minConnections === '' ? undefined : minConnections,
      maxConnections: maxConnections === '' ? undefined : maxConnections,
    });
  }, [propertyFilters, nodeTypes, edgeTypes, dateFrom, dateTo, minConnections, maxConnections, onFilterChange]);

  const handleClearAll = useCallback(() => {
    setNodeTypes([]);
    setEdgeTypes([]);
    setDateFrom(null);
    setDateTo(null);
    setMinConnections('');
    setMaxConnections('');
    setPropertyFilters({});
    onFilterChange({
      nodeTypes: [],
      edgeTypes: [],
      propertyFilters: {},
    });
  }, [onFilterChange]);

  const hasActiveFilters = nodeTypes.length > 0 || 
                           edgeTypes.length > 0 || 
                           dateFrom || 
                           dateTo || 
                           Object.keys(propertyFilters).length > 0 ||
                           minConnections !== '' ||
                           maxConnections !== '';

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <FilterListIcon />
          <Typography variant="h6">Filters</Typography>
        </Box>
        {hasActiveFilters && (
          <Button
            size="small"
            startIcon={<ClearIcon />}
            onClick={handleClearAll}
          >
            Clear All
          </Button>
        )}
      </Box>

      <Stack spacing={2}>
        {/* Node Type Filters */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Node Types
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {availableNodeTypes.map(type => (
              <Chip
                key={type}
                label={type}
                onClick={() => handleNodeTypeToggle(type)}
                color={nodeTypes.includes(type) ? 'primary' : 'default'}
                variant={nodeTypes.includes(type) ? 'filled' : 'outlined'}
              />
            ))}
          </Box>
        </Box>

        {/* Edge Type Filters */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Relationship Types
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {availableEdgeTypes.map(type => (
              <Chip
                key={type}
                label={type}
                onClick={() => handleEdgeTypeToggle(type)}
                color={edgeTypes.includes(type) ? 'primary' : 'default'}
                variant={edgeTypes.includes(type) ? 'filled' : 'outlined'}
              />
            ))}
          </Box>
        </Box>

        {/* Time-based Filtering */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Time Range
          </Typography>
          <Stack direction="row" spacing={2}>
            <TextField
              label="From Date"
              type="date"
              size="small"
              value={dateFrom ? dateFrom.toISOString().split('T')[0] : ''}
              onChange={(e) => {
                const date = e.target.value ? new Date(e.target.value) : null;
                handleDateChange(date, dateTo);
              }}
              InputLabelProps={{ shrink: true }}
              fullWidth
            />
            <TextField
              label="To Date"
              type="date"
              size="small"
              value={dateTo ? dateTo.toISOString().split('T')[0] : ''}
              onChange={(e) => {
                const date = e.target.value ? new Date(e.target.value) : null;
                handleDateChange(dateFrom, date);
              }}
              InputLabelProps={{ shrink: true }}
              fullWidth
            />
          </Stack>
        </Box>

        {/* Connection Count Filter */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Connection Count
          </Typography>
          <Stack direction="row" spacing={2}>
            <TextField
              label="Min Connections"
              type="number"
              size="small"
              value={minConnections}
              onChange={(e) => {
                const val = e.target.value === '' ? '' : parseInt(e.target.value);
                setMinConnections(val);
                onFilterChange({
                  nodeTypes,
                  edgeTypes,
                  dateFrom: dateFrom || undefined,
                  dateTo: dateTo || undefined,
                  propertyFilters,
                  minConnections: val === '' ? undefined : val,
                  maxConnections: maxConnections === '' ? undefined : maxConnections,
                });
              }}
              fullWidth
            />
            <TextField
              label="Max Connections"
              type="number"
              size="small"
              value={maxConnections}
              onChange={(e) => {
                const val = e.target.value === '' ? '' : parseInt(e.target.value);
                setMaxConnections(val);
                onFilterChange({
                  nodeTypes,
                  edgeTypes,
                  dateFrom: dateFrom || undefined,
                  dateTo: dateTo || undefined,
                  propertyFilters,
                  minConnections: minConnections === '' ? undefined : minConnections,
                  maxConnections: val === '' ? undefined : val,
                });
              }}
              fullWidth
            />
          </Stack>
        </Box>

        {/* Property Filters */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Property Filters
          </Typography>
          <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
            <TextField
              label="Property Key"
              size="small"
              value={propertyKey}
              onChange={(e) => setPropertyKey(e.target.value)}
              fullWidth
            />
            <TextField
              label="Property Value"
              size="small"
              value={propertyValue}
              onChange={(e) => setPropertyValue(e.target.value)}
              fullWidth
            />
            <Button
              variant="outlined"
              onClick={handleAddPropertyFilter}
              disabled={!propertyKey || !propertyValue}
            >
              Add
            </Button>
          </Stack>
          {Object.keys(propertyFilters).length > 0 && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {Object.entries(propertyFilters).map(([key, value]) => (
                <Chip
                  key={key}
                  label={`${key}: ${value}`}
                  onDelete={() => handleRemovePropertyFilter(key)}
                  color="secondary"
                />
              ))}
            </Box>
          )}
        </Box>
      </Stack>
    </Paper>
  );
}

