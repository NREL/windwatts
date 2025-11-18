import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  Button,
  Box,
  Typography,
  CircularProgress,
  Alert,
  Skeleton
} from "@mui/material";
import { useState, useContext } from "react";
import { DATA_MODEL_INFO } from "../../constants";
import { useDownloadCSVFile, useNearestGridLocation } from "../../hooks";
import { SettingsContext } from "../../providers/SettingsContext";
import { formatCoordinate } from "../../utils";

export const DownloadDialog = ({ onClose }: { onClose: () => void }) => {
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const [n_neighbors, setN_neighbors] = useState(1); // single nearest neighbor
  void setN_neighbors; // avoid unused variable warning
  
  const { canDownload, isDownloading, downloadFile, downloadBatchFiles } = useDownloadCSVFile();
  const {
    gridLocations,
    isLoading: isLoadingGridLocation,
    error: gridLocationError,
    retry: retryGridLocation,
  } = useNearestGridLocation(n_neighbors);

  const { currentPosition, preferredModel: dataModel } = useContext(SettingsContext);
  const { lat, lng } = currentPosition || {};
  const downloadInfo = dataModel ? DATA_MODEL_INFO[dataModel] : null;

  const nearestGridLocation = gridLocations.length > 0 ? gridLocations[0] : null;
  const hasError = !!(gridLocationError || downloadError);
  const isProcessing = isDownloading || isLoadingGridLocation;

  const hasEnoughNeighbors =
  (n_neighbors === 1 && gridLocations.length >= 1) ||
  (n_neighbors > 1 && gridLocations.length >= n_neighbors);

  const canConfirm = !isProcessing && !hasError && !!gridLocations && hasEnoughNeighbors;

  const handleConfirm = async () => {
    setDownloadError(null);

    if (gridLocations.length === 0) {
    setDownloadError("No nearest grid locations available.");
    return;
    }
    
    try {
      if (n_neighbors === 1 && nearestGridLocation) {
        const result = await downloadFile(
          nearestGridLocation.latitude,
          nearestGridLocation.longitude,
          nearestGridLocation.index
        );

        if (!result.success) {
          const errorMessage = result.error instanceof Error ? result.error.message : "Download failed";
          setDownloadError(errorMessage);
          return;
        }

        onClose();
      }
      else if (n_neighbors > 1) {
        const selection = gridLocations.slice(0, n_neighbors);
        if (selection.length < 2) {
          setDownloadError("Not enough neighbor grid points available for batch download.");
          return;
        }
        const result = await downloadBatchFiles(selection);
        if (!result.success) {
          const errorMessage = result.error instanceof Error ? result.error.message : "Batch download failed";
          setDownloadError(errorMessage);
          return;
        }

        onClose();
      }
    }
    catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unexpected error occurred";
      setDownloadError(errorMessage);
    }
  };

  const handleClose = () => {
    if (isProcessing) return;
    setDownloadError(null);
    onClose();
  };

  const handleRetry = () => {
    if (gridLocationError) {
      retryGridLocation();
    } else if (downloadError) {
      handleConfirm();
    }
  };

  const handleAction = hasError ? handleRetry : handleConfirm;

  return (
    <Dialog
      open={true}
      onClose={handleClose}
      aria-labelledby="download-dialog-title"
      aria-describedby="download-dialog-description"
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle id="download-dialog-title">
        Download Hourly Wind Data
        {isProcessing && <CircularProgress size={20} sx={{ ml: 2 }} />}
      </DialogTitle>
      <DialogContent>
        <DialogContentText id="download-dialog-description" component="div">
          
          {/* Missing data warning */}
          {!canDownload && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Missing required information</strong>
              </Typography>
              <Typography variant="body2" sx={{ mt: 1, fontSize: '0.85em' }}>
                Please select a location and data model from the settings before downloading.
              </Typography>
            </Alert>
          )}

          {/* Selected coordinates */}
          {canDownload && lat !== undefined && lng !== undefined && (
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Selected Coordinates:</strong> ({formatCoordinate(lat)}, {formatCoordinate(lng)})
            </Typography>
          )}
          
          {/* Grid location loading */}
          {isLoadingGridLocation && (
            <Box sx={{ mb: 1 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>Data Grid Coordinates:</strong>
              </Typography>
              <Skeleton variant="text" width="60%" height={20} />
              <Skeleton variant="text" width="80%" height={16} sx={{ mt: 0.5 }} />
            </Box>
          )}

          {/* Grid location error */}
          {gridLocationError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Failed to fetch grid location:</strong> {gridLocationError}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1, fontSize: '0.85em' }}>
                Please try again or contact support if the problem persists.
              </Typography>
            </Alert>
          )}

          {/* Download error */}
          {downloadError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Download failed:</strong> {downloadError}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1, fontSize: '0.85em' }}>
                Please try again.
              </Typography>
            </Alert>
          )}

          {/* Grid location success */}
          {nearestGridLocation && !isLoadingGridLocation && (
            <Box sx={{ mb: 1 }}>
              <Typography variant="body2">
                <strong>Data Grid Coordinates:</strong> ({formatCoordinate(nearestGridLocation.latitude)}, {formatCoordinate(nearestGridLocation.longitude)})
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontSize: '0.9em', 
                  color: '#666', 
                  fontStyle: 'italic',
                  mt: 0.5,
                  mb: 1
                }}
              >
                Note: Wind data is available at specific grid points. The download will contain data from the nearest grid point to your selected location.
              </Typography>
            </Box>
          )}

          {/* Data model information */}
          {isLoadingGridLocation ? (
            <Box>
              <Skeleton variant="text" width="70%" height={20} sx={{ mb: 1 }} />
              <Skeleton variant="text" width="50%" height={20} sx={{ mb: 1 }} />
              <Skeleton variant="text" width="90%" height={20} sx={{ mb: 1 }} />
              <Skeleton variant="text" width="85%" height={20} />
            </Box>
          ) : (
            downloadInfo && (
              <>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Data Source:</strong> {downloadInfo.description}
                </Typography>
                
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Year Range:</strong> {downloadInfo.year_range}
                </Typography>

                {downloadInfo.wind_speed_heights && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Windspeed Heights:</strong> {downloadInfo.wind_speed_heights.join(", ")}
                  </Typography>
                )}

                {downloadInfo.wind_direction_heights && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Wind Direction Heights:</strong> {downloadInfo.wind_direction_heights.join(", ")}
                  </Typography>
                )}
              </>
            )
          )}

          {/* Download progress */}
          {isDownloading && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <CircularProgress size={16} sx={{ mr: 1 }} />
                <Typography variant="body2">
                  Preparing your download... This may take a few moments.
                </Typography>
              </Box>
            </Alert>
          )}
        </DialogContentText>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button 
          onClick={handleClose}
          variant="outlined"
          sx={{ textTransform: "none" }}
        >
          Cancel
        </Button>
        <Button 
          onClick={handleAction}
          variant="contained"
          disabled={!canConfirm}
          sx={{ textTransform: "none" }}
        >
          {isDownloading ? 'Downloading...' :
          isLoadingGridLocation ? 'Loading...' : 
           hasError ? 'Retry' : 'Download'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};