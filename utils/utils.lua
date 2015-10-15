local function _flatten(flat_tt, tt)
        for _, item in pairs(tt) do
                if type(item) ~= 'table' then
                        table.insert(flat_tt, item)
                else
                        _flatten(flat_tt, item)
                end
        end
        return flat_tt
end

function flatten(tt)
        return _flatten({}, tt)
end

local function _extract_vocab(items, item_to_index, count)
        for _, item in ipairs(items) do
                if not item_to_index[item] then
                        count = count + 1
                        item_to_index[item] = count
                end
        end
        return item_to_index, count
end

function extract_vocab(items, ...)
        local item_to_index, index_to_item = {}, {}
        local count = 0
        if #{...} > 0 then
                item_to_index, count = _extract_vocab({...},
                                                item_to_index, count)
        end
        item_to_index = _extract_vocab(items, item_to_index, count)
        for item, index in pairs(item_to_index) do
                index_to_item[index] = item
        end
        return item_to_index, index_to_item
end
