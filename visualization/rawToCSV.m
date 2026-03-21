%{
    This script reads a TXT file and converts the data into a CSV file to
    make later analysis easier. The TXT file should be in the format:
        Lat:0.000000
        Lon:0.000000
        Sats:0
        Speed:0.00
        ...up to 
    with one attribute per line. For the above example the program below should be
    modifed so that:
        numFields = 4;
        labels = ["Lat","Lon","Sats","Speed"];
    to ensure proper data parsing. The fileName should also be changed to
    match the proper file, and the output file will automatically be named
    with the same name but with TXT replaced with CSV.
%}

%Parameters
fileName = "enoS2(1).TXT";
numFields = 17;
labels = ["Lat","Lon","Sats","Speed","GPS_Alt","Dist","Time","Alt","AX","AY","AZ","GX","GY","GZ","MX","MY","MZ"];

fileID = fopen(fileName,"r");
fileASCII = fread(fileID, '*char')';
fileContent = char(fileASCII.');
fileContent = string(fileContent.');
fclose(fileID);

fileContent = strjoin(fileContent,"");
fileContent = replace(fileContent,sprintf('\r\n'),",");
fileContent = replace(fileContent,sprintf('\r'),",");
fileContent = replace(fileContent,newline,",");
fileContent = split(fileContent,",");


data = {};
newEntry = false;
entry = {};


for i = 1:length(fileContent)
    ele = strtrim(fileContent(i));   % trim spaces
    
    if ele ~= ""                     % non-empty
        parts = split(ele, ":");     % split into label/value
        
        if numel(parts) == 2
            label = strtrim(parts(1));
            value = strtrim(parts(2));
            
            if label == labels(1)
                % Start new entry
                entry = {};
                newEntry = true;
                entry{end+1} = {label, value};
                
            elseif label == labels(end) && ~isempty(entry)
                % Close entry on Distance
                entry{end+1} = {label, value};
                newEntry = false;
                
            elseif newEntry && ~isempty(entry)
                % Collect other fields
                duplicate = false;
                for j = 1:length(entry)
                    if label == string(entry{j}{1})
                        duplicate = true;
                        break;
                    end
                end
                if  duplicate == false
                    entry{end+1} = {label, value};
                end
            end
            
            % When gathered all fields, save entry
            if newEntry == false && ~isempty(entry)
                data{end+1} = entry;
                entry = {};
            end
        end
    end
end

% Step 2: Extract numeric values for each entry
numEntries = length(data);
values = nan(numEntries,numFields);

for i = 1:numEntries
    loopLength = min([numFields,length(data{i})]);
    for j = 1:loopLength
        rawVal = string(data{i}{j}{2});
        % Remove any non-numeric parts (units, text)
        rawVal = regexprep(rawVal,'[^\d\.\-eE]',''); 
        for k = 1:length(labels)
            if data{i}{j}{1} == labels(k)
                values(i,k) = str2double(rawVal);
                break;
            end    
        end    
    end
end

% Step 3: Build table
T = array2table(values, 'VariableNames', cellstr(labels));

% Step 4: Write to CSV
name = fileName.replace('.txt','.csv').replace('.TXT','.csv');
writetable(T,name);
disp('CSV file written: '+name);
