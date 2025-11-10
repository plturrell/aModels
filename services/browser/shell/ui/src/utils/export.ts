/**
 * Data export utilities for charts and analytics
 */

/**
 * Export data to CSV format
 */
export function exportToCSV(data: any[], filename: string = 'export.csv'): void {
  if (!data || data.length === 0) {
    console.warn('No data to export');
    return;
  }

  // Get headers from first object
  const headers = Object.keys(data[0]);
  
  // Create CSV content
  const csvContent = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header];
        // Escape values containing commas, quotes, or newlines
        if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value ?? '';
      }).join(',')
    )
  ].join('\n');

  // Create blob and download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export data to JSON format
 */
export function exportToJSON(data: any, filename: string = 'export.json'): void {
  const jsonContent = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export chart as PNG image
 */
export async function exportChartToPNG(
  elementId: string,
  filename: string = 'chart.png',
  backgroundColor: string = '#ffffff'
): Promise<void> {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`Element with id ${elementId} not found`);
    return;
  }

  try {
    // Use html2canvas if available, otherwise fallback to canvas
    const html2canvas = (await import('html2canvas')).default;
    const canvas = await html2canvas(element, {
      backgroundColor,
      scale: 2, // Higher quality
      logging: false,
    });

    canvas.toBlob((blob) => {
      if (!blob) {
        console.error('Failed to create blob');
        return;
      }

      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }, 'image/png');
  } catch (error) {
    console.error('Error exporting chart to PNG:', error);
    // Fallback: try to get SVG and convert
    const svgElement = element.querySelector('svg');
    if (svgElement) {
      exportSVGToPNG(svgElement, filename);
    }
  }
}

/**
 * Export SVG to PNG
 */
function exportSVGToPNG(svgElement: SVGElement, filename: string): void {
  const svgData = new XMLSerializer().serializeToString(svgElement);
  const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) {
          const link = document.createElement('a');
          const downloadUrl = URL.createObjectURL(blob);
          link.setAttribute('href', downloadUrl);
          link.setAttribute('download', filename);
          link.style.visibility = 'hidden';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(downloadUrl);
        }
      }, 'image/png');
    }
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

/**
 * Export chart as SVG
 */
export function exportChartToSVG(elementId: string, filename: string = 'chart.svg'): void {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`Element with id ${elementId} not found`);
    return;
  }

  const svgElement = element.querySelector('svg');
  if (!svgElement) {
    console.error('No SVG element found in chart');
    return;
  }

  const svgData = new XMLSerializer().serializeToString(svgElement);
  const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(svgBlob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export dashboard as PDF (requires jsPDF)
 */
export async function exportToPDF(
  elementId: string,
  filename: string = 'dashboard.pdf',
  title?: string
): Promise<void> {
  try {
    const html2canvas = (await import('html2canvas')).default;
    const jsPDF = (await import('jspdf')).default;

    const element = document.getElementById(elementId);
    if (!element) {
      console.error(`Element with id ${elementId} not found`);
      return;
    }

    const canvas = await html2canvas(element, {
      scale: 2,
      logging: false,
    });

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF.jsPDF('p', 'mm', 'a4');
    
    if (title) {
      pdf.setFontSize(16);
      pdf.text(title, 10, 10);
    }

    const imgWidth = 210; // A4 width in mm
    const pageHeight = 295; // A4 height in mm
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    let heightLeft = imgHeight;
    let position = title ? 20 : 0;

    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
    heightLeft -= pageHeight;

    while (heightLeft >= 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;
    }

    pdf.save(filename);
  } catch (error) {
    console.error('Error exporting to PDF:', error);
    // Fallback to PNG
    exportChartToPNG(elementId, filename.replace('.pdf', '.png'));
  }
}

/**
 * Share dashboard link with state
 */
export function shareDashboardLink(state: Record<string, any>): string {
  const baseUrl = window.location.origin + window.location.pathname;
  const encodedState = btoa(JSON.stringify(state));
  return `${baseUrl}?state=${encodedState}`;
}

/**
 * Copy shareable link to clipboard
 */
export async function copyShareableLink(state: Record<string, any>): Promise<boolean> {
  try {
    const link = shareDashboardLink(state);
    await navigator.clipboard.writeText(link);
    return true;
  } catch (error) {
    console.error('Error copying link to clipboard:', error);
    return false;
  }
}

