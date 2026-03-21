%{
    This script reads a CSV(spreadsheet) file and maps out the speed
    of the sensor at each point. Make sure to set the key
    parameters listed below to get the final figure that matches your
    desired data.
%}

%Parameters
fileName = 'enoFeb16th_smoothed.csv';
useCoordinatePositioning = false;       % true = use lat and long instead of meters for axes
plotAltitude = false;                   % this makes the plot either 2D or 3D
figRot = 15;                            % specify an angle in degrees to rotate coordinate system(CCW and only for meter-based coord system)
useSatelliteBackground = true;          % specifies whether to display satellite imagery behind data

createGridMap = false; %only works in 2D
overlayDataPoints = true;
% Choose grid resolution:
nx = 10;   % number of bins in x
ny = 15;   % number of bins in y

% Read CSV
T = readtable(fileName);
% --- Filter valid rows ---
validIdx = (T.Lat ~= 0) | (T.Lon ~= 0);
T = T(validIdx, :);

% Extract variables
time = T.Time;
lat  = T.Lat;
lon  = T.Lon;
alt  = T.GPS_Alt;   % gps altitude reading(meters)
%alt = T.Alt; %accelerometer altitude reading(meters from starting pos)
spd  = T.Speed; %initially in miles per hour
spd = spd./2.237; %divide by 2.237 to get speed in m/s

%fill in gaps in every other GPS altitude point(every other one is zero)
%alt(alt == 0) = interp1(find(alt ~= 0), alt(alt ~= 0), find(alt == 0), 'linear', 'extrap');

if(useCoordinatePositioning)
    %use raw longitude and latitude values for axes
    x = lon;
    y = lat;
    xName = 'Longitude';
    yName  = 'Latitude';
else
    % --- Convert lat/lon to local Cartesian meters ---
    R = 6371000; % Earth radius in meters
    lat0 = deg2rad(lat(1));
    lon0 = deg2rad(lon(1));
    x = R * (deg2rad(lon) - lon0) .* cos(lat0); % Easting
    y = R * (deg2rad(lat) - lat0);              % Northing
    xRot = x*cos(deg2rad(figRot)) - y*sin(deg2rad(figRot));
    yRot = x*sin(deg2rad(figRot)) + y*cos(deg2rad(figRot));
    x = xRot;
    y = yRot;
    if(figRot == 0)
        xName = 'East [m]';
        yName  = 'North [m]';
    else
        xName = 'x [m]';
        yName = 'y [m]';
    end
end
z = alt;                                    % Altitude (m)

% --- Normalize speed colormap ---
spd = double(spd);
spdMin = min(spd);
spdMax = max(spd);

if(plotAltitude)
    % --- Plot 3D trajectory ---
    figure;
    scatter3(x, y, z, 50, spd, 'filled'); % color by speed
    hold on;
    %plot3(x, y, z, 'k-', 'LineWidth', 1); % connect points with a line(useful for real-life use cases)
    plot3(x, y, z,'LineStyle','none'); %no lines between data points(useful for sensors that traversed route multiple times)
    
    xlabel(xName);
    ylabel(yName);
    zlabel('Altitude (m)');
    title('3D GPS Trajectory (colored by Speed)');
    colormap(turbo);
    cb = colorbar;
    ylabel(cb, 'Speed(m/s)');
    clim([spdMin spdMax]);
    grid on;
    if(~useCoordinatePositioning)
        axis equal;
    end
    view(3);
else
    if(createGridMap)
        % Build edge values
        xedges = linspace(min(x), max(x), nx+1);
        yedges = linspace(min(y), max(y), ny+1);
        
        % Bin each point
        ix = discretize(x, xedges);   % returns 1..nx
        iy = discretize(y, yedges);   % returns 1..ny
        
        % Keep only points that fell inside the grid
        valid = ~isnan(ix) & ~isnan(iy) & ~isnan(spd);
        ix = ix(valid);
        iy = iy(valid);
        s = spd(valid);
        % Linear index for accumarray: rows = iy (y dir), cols = ix (x dir)
        linIdx = sub2ind([ny nx], iy, ix);
        
        % Sum of values and counts per bin
        sumVals  = accumarray(linIdx, s, [ny*nx, 1], @sum, NaN);
        countVals = accumarray(linIdx, 1, [ny*nx, 1], @sum, 0);
        
        % Mean per bin (NaN where count==0)
        meanValsVec = sumVals ./ max(countVals,1); % avoid divide-by-zero
        meanValsVec(countVals==0) = NaN;
        
        % Reshape into 2D matrix (ny rows, nx cols)
        meanVals = reshape(meanValsVec, ny, nx);
        
        % Compute bin centers for axes
        xcenters = (xedges(1:end-1) + xedges(2:end))/2;
        ycenters = (yedges(1:end-1) + yedges(2:end))/2;
        
        % Plot as blocks. imagesc will show each matrix element as a colored block.
        figure;
        imagesc(xcenters, ycenters, meanVals); 

        h = imagesc(xcenters, ycenters, meanVals);
        set(gca,'YDir','normal')
        colormap(turbo);
        cb = colorbar;
        ylabel(cb, 'Speed [m s^{-1}]', 'FontSize',12);
        clim([spdMin spdMax]);
        % Make NaNs(grid spots with no data) transparent
        set(h,'AlphaData',~isnan(meanVals))

        xlabel(xName, 'FontSize',12);
        ylabel(yName, 'FontSize',12);
        %title('Binned mean speed values (grid)');
        title('');
        xlim([min(xcenters)-(max(x)-min(x))/nx, max(xcenters)]);
        axis equal;
        axis tight;
        if(overlayDataPoints)
            hold on;
            plot(x, y, '.k', 'MarkerSize', 6);
            hold off;
        end
        view(2);
    else
        figure;
        % --- Plot 2D trajectory ---
        if(useSatelliteBackground)
            gx = geoaxes;
            geobasemap(gx,'satellite');
            hold on;
            geoscatter(gx, lat, lon, 50, spd, 'filled');
            colormap(turbo);
            cb = colorbar;
            ylabel(cb, 'Sensor Speed [m/s]');
            grid on;
            view(2);
        else
            scatter(x, y, 50, spd, 'filled'); % color by speed
            hold on;
            %plot(x, y, 'k-', 'LineWidth', 1); % connect points with a line(useful for real-life use cases)
            plot(x, y,'LineStyle','none'); %no lines between data points(useful for sensors that traversed route multiple times)
            xlabel(xName);
            ylabel(yName);
            title('2D GPS Trajectory (colored by Speed)');
            colormap(turbo);
            cb = colorbar;
            ylabel(cb, 'Speed [m/s]');
            clim([spdMin spdMax]);
            grid on;
            if(~useCoordinatePositioning)
                axis equal;
            end
            view(2);
        end
    end
end