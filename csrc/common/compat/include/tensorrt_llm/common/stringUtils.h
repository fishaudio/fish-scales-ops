/*
 * Standalone shim for tensorrt_llm/common/stringUtils.h
 *
 * Provides only fmtstr, used by TLLM_THROW / TLLM_CHECK_WITH_INFO.
 */
#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

inline std::string fmtstr(std::string const& s)
{
    return s;
}

inline std::string fmtstr(std::string&& s)
{
    return std::move(s);
}

inline std::string fmtstr(char const* fmt, ...)
{
    if (fmt == nullptr)
    {
        return std::string{};
    }
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int needed = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    if (needed < 0)
    {
        va_end(ap2);
        return std::string{"<fmtstr error>"};
    }
    std::vector<char> buf(static_cast<size_t>(needed) + 1u);
    std::vsnprintf(buf.data(), buf.size(), fmt, ap2);
    va_end(ap2);
    return std::string(buf.data(), static_cast<size_t>(needed));
}

} // namespace common
TRTLLM_NAMESPACE_END
