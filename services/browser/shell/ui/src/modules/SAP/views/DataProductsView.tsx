/**
 * SAP Data Products View
 * 
 * Browse and explore available SAP BDC data products
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Stack,
  Alert,
  CircularProgress,
  Chip,
  TextField,
  InputAdornment,
} from '@mui/material';
import { GridLegacy as Grid } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import RefreshIcon from '@mui/icons-material/Refresh';
import InfoIcon from '@mui/icons-material/Info';
import { Panel } from '../../../components/Panel';
import { useApiData } from '../../../api/client';
import { listSAPDataProducts, listSAPIntelligentApplications } from '../../../api/sap';
import type { SAPBDCConnectionConfig, SAPDataProduct, SAPIntelligentApplication } from '../../../api/sap';

interface DataProductsViewProps {
  connection: SAPBDCConnectionConfig;
}

export function DataProductsView({ connection }: DataProductsViewProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedProduct, setSelectedProduct] = useState<SAPDataProduct | null>(null);

  const {
    data: dataProducts,
    loading: productsLoading,
    error: productsError,
    refresh: refreshProducts,
  } = useApiData<SAPDataProduct[], SAPDataProduct[]>(
    '/v2/sap-bdc/data-products',
    undefined,
    []
  );

  const {
    data: intelligentApps,
    loading: appsLoading,
    error: appsError,
    refresh: refreshApps,
  } = useApiData<SAPIntelligentApplication[], SAPIntelligentApplication[]>(
    '/sap-bdc/intelligent-applications',
    undefined,
    []
  );

  const filteredProducts = dataProducts?.filter((product) =>
    product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    product.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    product.description?.toLowerCase().includes(searchQuery.toLowerCase())
  ) || [];

  const handleProductSelect = (product: SAPDataProduct) => {
    setSelectedProduct(product);
  };

  if (productsLoading || appsLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  if (productsError) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {productsError.message || 'Failed to load data products. Make sure the SAP BDC service is configured and running.'}
      </Alert>
    );
  }

  return (
    <Stack spacing={3}>
      <Panel
        title="Data Products"
        subtitle="Browse available data products from SAP Business Data Cloud"
        dense
        actions={
          <Button
            startIcon={<RefreshIcon />}
            onClick={() => {
              refreshProducts();
              refreshApps();
            }}
            size="small"
          >
            Refresh
          </Button>
        }
      >
        <TextField
          fullWidth
          placeholder="Search data products..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          sx={{ mb: 2 }}
        />

        {filteredProducts.length === 0 ? (
          <Alert severity="info" icon={<InfoIcon />}>
            {searchQuery
              ? 'No data products match your search query.'
              : 'No data products available. Make sure your connection is configured correctly.'}
          </Alert>
        ) : (
          <Grid container spacing={2}>
            {filteredProducts.map((product) => (
              <Grid item xs={12} sm={6} md={4} key={product.id}>
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    cursor: 'pointer',
                    '&:hover': {
                      boxShadow: 4,
                    },
                  }}
                  onClick={() => handleProductSelect(product)}
                >
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      {product.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {product.description || 'No description available'}
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      <Chip label={`ID: ${product.id}`} size="small" />
                      {product.version && (
                        <Chip label={`v${product.version}`} size="small" variant="outlined" />
                      )}
                      {product.status && (
                        <Chip
                          label={product.status}
                          size="small"
                          color={product.status === 'active' ? 'success' : 'default'}
                        />
                      )}
                    </Stack>
                  </CardContent>
                  <CardActions>
                    <Button size="small" onClick={() => handleProductSelect(product)}>
                      View Details
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Panel>

      {intelligentApps && intelligentApps.length > 0 && (
        <Panel title="Intelligent Applications" dense>
          <Grid container spacing={2}>
            {intelligentApps.map((app) => (
              <Grid item xs={12} sm={6} md={4} key={app.id}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {app.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {app.description || 'No description available'}
                    </Typography>
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      <Chip label={app.type || 'Application'} size="small" />
                      {app.status && (
                        <Chip
                          label={app.status}
                          size="small"
                          color={app.status === 'active' ? 'success' : 'default'}
                        />
                      )}
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Panel>
      )}

      {selectedProduct && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {selectedProduct.name}
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              {selectedProduct.description || 'No description available'}
            </Typography>
            <Typography variant="caption" display="block" sx={{ mb: 1 }}>
              <strong>ID:</strong> {selectedProduct.id}
            </Typography>
            {selectedProduct.version && (
              <Typography variant="caption" display="block" sx={{ mb: 1 }}>
                <strong>Version:</strong> {selectedProduct.version}
              </Typography>
            )}
            {selectedProduct.metadata && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Metadata:
                </Typography>
                <pre style={{ fontSize: '0.75rem', overflow: 'auto' }}>
                  {JSON.stringify(selectedProduct.metadata, null, 2)}
                </pre>
              </Box>
            )}
          </CardContent>
        </Card>
      )}
    </Stack>
  );
}


