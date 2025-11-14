/**
 * GraphFilters Component
 * 
 * Filters for graph visualization
 */

import React from 'react';
import { Box, Typography, FormControl, InputLabel, Select, MenuItem, Checkbox, ListItemText, OutlinedInput } from '@mui/material';

export interface GraphFilterState {
  nodeTypes: string[];
  edgeTypes: string[];
  propertyFilters: Record<string, any>;
}

interface GraphFiltersProps {
  availableNodeTypes: string[];
  availableEdgeTypes: string[];
  onFilterChange: (filters: GraphFilterState) => void;
  initialFilters?: GraphFilterState;
}

export function GraphFilters({
  availableNodeTypes,
  availableEdgeTypes,
  onFilterChange,
  initialFilters = { nodeTypes: [], edgeTypes: [], propertyFilters: {} }
}: GraphFiltersProps) {
  const [filters, setFilters] = React.useState(initialFilters);

  const handleNodeTypeChange = (event: any) => {
    const value = event.target.value;
    const newFilters = { ...filters, nodeTypes: value };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const handleEdgeTypeChange = (event: any) => {
    const value = event.target.value;
    const newFilters = { ...filters, edgeTypes: value };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Filters
      </Typography>
      
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Node Types</InputLabel>
        <Select
          multiple
          value={filters.nodeTypes}
          onChange={handleNodeTypeChange}
          input={<OutlinedInput label="Node Types" />}
          renderValue={(selected) => selected.join(', ')}
        >
          {availableNodeTypes.map((type) => (
            <MenuItem key={type} value={type}>
              <Checkbox checked={filters.nodeTypes.indexOf(type) > -1} />
              <ListItemText primary={type} />
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControl fullWidth>
        <InputLabel>Edge Types</InputLabel>
        <Select
          multiple
          value={filters.edgeTypes}
          onChange={handleEdgeTypeChange}
          input={<OutlinedInput label="Edge Types" />}
          renderValue={(selected) => selected.join(', ')}
        >
          {availableEdgeTypes.map((type) => (
            <MenuItem key={type} value={type}>
              <Checkbox checked={filters.edgeTypes.indexOf(type) > -1} />
              <ListItemText primary={type} />
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </Box>
  );
}
