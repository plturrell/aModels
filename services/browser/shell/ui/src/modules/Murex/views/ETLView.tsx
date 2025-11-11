/**
 * Murex ETL View
 * Displays ETL transformation information and SAP GL integration
 */

import React from "react";
import {
  Box,
  Typography,
  Stack,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow
} from "@mui/material";
import { Panel } from "../../../components/Panel";

export function ETLView() {
  return (
    <Stack spacing={2}>
      <Panel title="ETL Pipeline" dense>
        <Stack spacing={2}>
          <Typography variant="body2" color="text.secondary">
            The Murex ETL pipeline transforms trade data into SAP GL journal entries:
          </Typography>
          
          <Stack spacing={2} mt={2}>
            <Box display="flex" alignItems="center" gap={2}>
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: "50%",
                  bgcolor: "primary.main",
                  color: "white",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: 600
                }}
              >
                1
              </Box>
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  Murex API
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Extract trades from Murex
                </Typography>
              </Box>
            </Box>

            <Box display="flex" alignItems="center" gap={2}>
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: "50%",
                  bgcolor: "success.main",
                  color: "white",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: 600
                }}
              >
                2
              </Box>
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  Transform
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Map to SAP GL format
                </Typography>
              </Box>
            </Box>

            <Box display="flex" alignItems="center" gap={2}>
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: "50%",
                  bgcolor: "warning.main",
                  color: "white",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: 600
                }}
              >
                3
              </Box>
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  SAP GL
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Load journal entries
                </Typography>
              </Box>
            </Box>
          </Stack>
        </Stack>
      </Panel>

      <Panel title="Field Mappings" dense>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Murex Field</strong></TableCell>
              <TableCell><strong>SAP GL Field</strong></TableCell>
              <TableCell><strong>Transformation</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell>trade_id</TableCell>
              <TableCell>entry_id</TableCell>
              <TableCell>JE-{`{trade_id}`}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>trade_date</TableCell>
              <TableCell>entry_date</TableCell>
              <TableCell>Identity</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>notional_amount</TableCell>
              <TableCell>debit_amount</TableCell>
              <TableCell>Identity</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>notional_amount</TableCell>
              <TableCell>credit_amount</TableCell>
              <TableCell>Copy</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>counterparty_id</TableCell>
              <TableCell>account</TableCell>
              <TableCell>Lookup</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </Panel>
    </Stack>
  );
}

