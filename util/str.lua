-- Use the pattern `pt` to split the string `str`
-- pt: [str] define which part of `str` will be reserve.
-- skip: [boolean] if true, will skip the blank part. *default* is true.
local function split(str, pt, skip)
        skip = skip or true
        local ret = {}
        if skip then
                for chunk in string.gfind(str, pt) do
                        table.insert(ret, chunk)
                end
        else
                for chunk in string.gfind(str,pt) do
                        if chunk ~= '' then
                                table.insert(ret, chunk)
                        end
                end
        end
        return ret
end

function split_line(str, skip)
        return split(str, '(.-)\n', skip or true)
end

function split_word(str)
        return split(str, '%S+')
end
