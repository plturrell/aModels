import type { PropsWithChildren, ReactNode } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  useTheme
} from '@mui/material';

interface PanelProps {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  dense?: boolean;
}

export function Panel({ title, subtitle, actions, dense, children }: PropsWithChildren<PanelProps>) {
  const theme = useTheme();
  
  return (
    <Card 
      sx={{ 
        mb: dense ? 2 : 3,
        boxShadow: theme.shadows[2]
      }}
    >
      <CardHeader
        title={
          <Typography variant="h6" component="h2">
            {title}
          </Typography>
        }
        subheader={
          subtitle ? (
            <Typography variant="body2" color="text.secondary" component="div">
              {subtitle}
            </Typography>
          ) : undefined
        }
        action={actions ? <Box sx={{ display: 'flex', alignItems: 'center' }}>{actions}</Box> : undefined}
        sx={{
          pb: dense ? 1 : 2,
          '& .MuiCardHeader-action': {
            margin: 0,
            alignSelf: 'center'
          }
        }}
      />
      <CardContent sx={{ pt: dense ? 1 : 2 }}>
        {children}
      </CardContent>
    </Card>
  );
}
